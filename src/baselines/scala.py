"""SCALA-NIDS adaptation: chunk prediction precedes unlabeled target updating."""
from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch import nn
from scipy.stats import chisquare
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class Detector(nn.Module):
    def __init__(self, dimensions, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimensions, hidden[0]), nn.ReLU(),
                                     nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.classifier = nn.Linear(hidden[1], 2)
        self.decoder = nn.Sequential(nn.Linear(hidden[1], hidden[0]), nn.ReLU(),
                                     nn.Linear(hidden[0], dimensions))
        self.domain = nn.Sequential(nn.Linear(hidden[1], 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, values):
        latent = self.encoder(values)
        return latent, self.classifier(latent), self.decoder(latent)


def fallback_strategy(snapshot):
    signals = snapshot['signals']
    active = sum(bool(signals[key]) for key in ('domain', 'confidence', 'distribution'))
    if active > 1:
        return 'mixed'
    if signals['domain'] or signals['distribution']:
        return 'covariate'
    if snapshot['unknown_ratio'] > .20:
        return 'label'
    if signals['confidence']:
        return 'concept'
    return 'conservative'


def validate_advice(advice, snapshot):
    signals = snapshot['signals']
    supported = {
        'covariate': signals['domain'] or signals['distribution'],
        'concept': signals['confidence'],
        'label': snapshot['unknown_ratio'] > .20,
        'mixed': sum(bool(value) for value in signals.values()) >= 2}
    evidence = any(supported.values())
    drift = advice['drift_type']
    if not ((drift == 'none' and not evidence) or supported.get(drift, False)):
        return False
    if advice['adapt'] and not evidence:
        return False
    # A conservative response is allowed even in a supported drift setting.
    return advice['strategy'] == 'conservative' or advice['strategy'] == drift


class SCALANIDS:
    def __init__(self, reasoner, options, source, labels, validation, validation_labels,
                 numeric_dimensions, seed):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.reasoner, self.options = reasoner, options
        self.source, self.labels = torch.as_tensor(source), torch.as_tensor(labels, dtype=torch.long)
        self.validation = torch.as_tensor(validation)
        self.validation_labels = torch.as_tensor(validation_labels, dtype=torch.long)
        self.model = Detector(source.shape[1], options['hidden'])
        self.weights = torch.tensor([3.] * numeric_dimensions +
                                    [.3] * (source.shape[1] - numeric_dimensions))
        self.buffer = torch.empty((0, source.shape[1]))
        self.chunk_index = self.accepted = self.rejected = self.fallbacks = 0
        self.events = []

    def batches(self, count):
        order = self.rng.permutation(count)
        for start in range(0, count, self.options['batch_size']):
            yield order[start:start + self.options['batch_size']]

    def fit(self, directory):
        if not (self.labels == 0).any() or not (self.labels == 1).any():
            raise ValueError('SCALA training requires both benign and attack examples')
        # Stage 1: frozen encoder, domain-head warm-up on perturbed source features.
        optimizer = torch.optim.Adam(self.model.domain.parameters(), lr=self.options['learning_rate'])
        for _ in range(self.options['domain_epochs']):
            for indices in self.batches(len(self.source)):
                x = self.source[indices]
                with torch.no_grad():
                    latent = self.model.encoder(torch.cat([x, x + .1 * torch.randn_like(x)]))
                target = torch.cat([torch.zeros(len(x)), torch.ones(len(x))]).long()
                loss = nn.functional.cross_entropy(self.model.domain(latent), target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Stage 2: benign-only weighted reconstruction.
        optimizer = torch.optim.Adam(list(self.model.encoder.parameters()) +
                                     list(self.model.decoder.parameters()), lr=self.options['learning_rate'])
        benign = self.source[self.labels == 0]
        for _ in range(self.options['reconstruction_epochs']):
            for indices in self.batches(len(benign)):
                x = benign[indices]
                reconstruction = self.model(x)[2]
                loss = (((reconstruction - x) ** 2) * self.weights).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Stage 3: weighted joint objective, detached classifier training, EMA center.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options['learning_rate'])
        center = None
        for epoch in range(self.options['joint_epochs']):
            for indices in self.batches(len(self.source)):
                x, y = self.source[indices], self.labels[indices]
                z, _, reconstruction = self.model(x)
                normal = y == 0
                recon = (((reconstruction[normal] - x[normal]) ** 2) * self.weights).mean() if normal.any() else z.sum() * 0
                if normal.any():
                    batch_center = z[normal].mean(0).detach()
                    center = batch_center if center is None else .9 * center + .1 * batch_center
                compact = ((z[normal] - center) ** 2).mean() if normal.any() else z.sum() * 0
                class_counts = torch.bincount(y, minlength=2).float().clamp_min(1)
                ce = nn.functional.cross_entropy(self.model.classifier(z.detach()), y,
                                                 weight=1 / class_counts)
                shifted = self.model.encoder(x + .1 * torch.randn_like(x))
                domain = self.model.domain(Reverse.apply(torch.cat([z, shifted])))
                domain_target = torch.cat([torch.zeros(len(x)), torch.ones(len(x))]).long()
                domain_loss = nn.functional.cross_entropy(domain, domain_target)
                loss = recon + .4 * compact + .3 * ce + .1 * min(1., (epoch + 1) / 10) * domain_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.model.eval()
        with torch.no_grad():
            errors = self.measure(self.model, benign)['reconstruction_error']
            self.reconstruction_threshold = max(float(np.quantile(errors, self.options['reconstruction_quantile'])), 1e-6)
            probabilities = self.measure(self.model, self.source)['probabilities']
            self.reference_distribution = np.bincount(probabilities.argmax(1), minlength=2) + 1.
            self.reference_distribution /= self.reference_distribution.sum()
        # Fixed source retention buffer; calibration/validation never enters this buffer.
        selected = self.rng.permutation(len(self.source))[:self.options['buffer_size']]
        self.source, self.labels = self.source[selected], self.labels[selected]
        self.save(directory)

    @staticmethod
    def measure(model, values):
        with torch.no_grad():
            z, logits, reconstruction = model(values)
            probabilities = logits.softmax(1).numpy()
            errors = ((reconstruction - values) ** 2).mean(1).numpy()
        return {'probabilities': probabilities, 'reconstruction_error': errors,
                'latent': z.numpy()}

    def scores(self, model, values):
        result = self.measure(model, values)
        p = result['probabilities']
        result['confidence'] = p.max(1)
        result['entropy'] = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
        # Paper leaves u's exact definition unspecified; use normalized reconstruction error.
        result['unknownness'] = result['reconstruction_error'] / self.reconstruction_threshold
        alpha = self.options['anomaly_alpha']
        result['anomaly'] = alpha * result['unknownness'] + (1 - alpha) * p[:, 1]
        return result

    def snapshot(self, values, measured):
        source_latent = self.measure(self.model, self.source)['latent']
        target_latent = measured['latent']
        n = min(len(source_latent), len(target_latent), self.options['buffer_size'])
        domain_accuracy = .5
        if n >= 4:
            x = np.concatenate([source_latent[:n], target_latent[:n]])
            y = np.repeat([0, 1], n)
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.4,
                                                                             stratify=y, random_state=17)
            probe = LogisticRegression(max_iter=300).fit(train_x, train_y)
            domain_accuracy = float(probe.score(test_x, test_y))
        counts = np.bincount(measured['probabilities'].argmax(1), minlength=2)
        pvalue = float(chisquare(counts, self.reference_distribution * len(values)).pvalue)
        return {'chunk': self.chunk_index, 'flows': len(values),
                'mean_confidence': float(measured['confidence'].mean()),
                'predicted_distribution': (counts / len(values)).tolist(),
                'domain_accuracy': domain_accuracy, 'distribution_pvalue': pvalue,
                'unknown_ratio': float((measured['unknownness'] > self.options['unknown_threshold']).mean()),
                'buffer_size': len(self.buffer), 'accepted_updates': self.accepted,
                'signals': {'domain': domain_accuracy > self.options['domain_threshold'],
                            'confidence': float(measured['confidence'].mean()) < self.options['confidence_threshold'],
                            'distribution': pvalue < self.options['distribution_pvalue']}}

    def advise(self, snapshot):
        fallback = fallback_strategy(snapshot)
        try:
            result = self.reasoner.request('adaptation_strategy',
                'Advise adaptation using aggregate statistics only. Do not generate parameters. '
                'Domain/distribution signals support covariate drift; confidence degradation supports '
                'concept drift; unknown-ratio above 0.20 supports label drift; multiple active '
                'signals support mixed drift. Choose a supported drift type and its matching '
                'strategy, or a conservative response. Use conservative/none without evidence. '
                'Periodic inspection alone does not justify adaptation.', snapshot,
                {'adapt': {'type': 'boolean'},
                 'drift_type': {'type': 'string', 'enum': ['none', 'covariate', 'label', 'concept', 'mixed']},
                 'strategy': {'type': 'string', 'enum': list(self.options['templates'])}})
            if not validate_advice(result, snapshot):
                raise ValueError('semantically_inconsistent_advice')
            return {**result, 'source': 'llm'}
        except (ValueError, KeyError, TypeError, IndexError):
            self.fallbacks += 1
            return {'adapt': any(snapshot['signals'].values()) or snapshot['unknown_ratio'] > .20, 'strategy': fallback,
                    'drift_type': 'none' if fallback == 'conservative' else fallback,
                    'source': 'fallback'}

    def objective(self, student, teacher, source, labels, target, template, *, perturb=True):
        zs, logits, _ = student(source)
        zt, target_logits, _ = student(target)
        with torch.no_grad():
            _, teacher_logits, _ = teacher(target)
            teacher_prob = teacher_logits.softmax(1)
        confidence, pseudo = teacher_prob.max(1)
        mask = confidence > template['pseudo_threshold']
        src = nn.functional.cross_entropy(logits, labels)
        pl = nn.functional.cross_entropy(target_logits[mask], pseudo[mask]) if mask.any() else target_logits.sum() * 0
        student_input = target + .01 * torch.randn_like(target) if perturb else target
        teacher_input = target + .01 * torch.randn_like(target) if perturb else target
        with torch.no_grad():
            stable = teacher(teacher_input)[1].softmax(1)
        consistency = ((student(student_input)[1].softmax(1) - stable) ** 2).sum(1).mean()
        domain = student.domain(Reverse.apply(torch.cat([zs, zt])))
        domain_labels = torch.cat([torch.zeros(len(zs)), torch.ones(len(zt))]).long()
        alignment = nn.functional.cross_entropy(domain, domain_labels)
        terms = [src, alignment, pl, consistency]
        return sum(weight * term for weight, term in zip(template['weights'], terms))

    def adapt(self, strategy):
        template = self.options['templates'][strategy]
        teacher, student = copy.deepcopy(self.model), copy.deepcopy(self.model)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        for parameter in student.encoder[0].parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()),
                                     lr=template['learning_rate'])
        source, labels, target = self.source, self.labels, self.buffer
        with torch.no_grad():
            loss_before = float(self.objective(teacher, teacher, source, labels, target, template, perturb=False))
        ratio_s, ratio_t = template['source_target_ratio']
        target_batch = max(1, round(self.options['batch_size'] * ratio_t / (ratio_s + ratio_t)))
        source_batch = max(1, self.options['batch_size'] - target_batch)
        for _ in range(template['epochs']):
            for _ in range(max(1, int(np.ceil(len(target) / target_batch)))):
                s = self.rng.integers(len(source), size=source_batch)
                t = self.rng.integers(len(target), size=target_batch)
                loss = self.objective(student, teacher, source[s], labels[s], target[t], template)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), self.options['gradient_clip'])
                optimizer.step()
        student.eval()
        with torch.no_grad():
            after_loss = float(self.objective(student, teacher, source, labels, target, template, perturb=False))
        old_validation = self.scores(teacher, self.validation)['anomaly'] > self.options['anomaly_threshold']
        new_validation = self.scores(student, self.validation)['anomaly'] > self.options['anomaly_threshold']
        truth = self.validation_labels.numpy()
        before, after = self.scores(teacher, target), self.scores(student, target)
        distribution = np.bincount(after['probabilities'].argmax(1), minlength=2) / len(target)
        entropy = float(-(distribution * np.log(np.maximum(distribution, 1e-12))).sum())
        checks = {
            'loss_stability': bool(np.isfinite(after_loss) and after_loss / max(loss_before, 1e-8) <= self.options['loss_ratio_max']),
            'source_retention': bool((new_validation == truth).mean() >= (old_validation == truth).mean() - self.options['source_accuracy_tolerance']),
            'target_confidence': bool(after['confidence'].mean() >= before['confidence'].mean() - self.options['confidence_tolerance']),
            'noncollapse': bool(entropy > self.options['prediction_entropy_min']),
            'unknownness_stability': bool(abs((after['unknownness'] > self.options['unknown_threshold']).mean() -
                                             (before['unknownness'] > self.options['unknown_threshold']).mean()) < self.options['unknown_rate_tolerance'])}
        if all(checks.values()):
            self.model = student
            self.accepted += 1
        else:
            self.rejected += 1
        return {'accepted': all(checks.values()), 'checks': checks,
                'loss_before': loss_before, 'loss_after': after_loss}

    def predict_chunk(self, values):
        self.chunk_index += 1
        values = torch.as_tensor(values)
        measured = self.scores(self.model, values)
        # Save every prediction before using any current-chunk features for adaptation.
        predictions = [{'label': 'ATTACK' if score > self.options['anomaly_threshold'] else 'BENIGN',
                        'anomaly_score': float(score), 'classifier_confidence': float(confidence)}
                       for score, confidence in zip(measured['anomaly'], measured['confidence'])]
        snapshot = self.snapshot(values, measured)
        safe = ((measured['unknownness'] < self.options['buffer_unknown_max']) &
                (measured['confidence'] > self.options['buffer_confidence_min']) &
                (measured['entropy'] < self.options['buffer_entropy_max']))
        self.buffer = torch.cat([self.buffer, values[safe]])[-self.options['buffer_size']:]
        inspect = any(snapshot['signals'].values()) or self.chunk_index % self.options['inspection_period'] == 0
        event = {'snapshot': snapshot, 'inspected': inspect, 'safe_buffer_size': len(self.buffer)}
        if inspect:
            advice = self.advise(snapshot)
            event['advice'] = advice
            if advice['adapt'] and len(self.buffer):
                event['update'] = self.adapt(advice['strategy'])
        self.events.append(event)
        return predictions

    def save(self, directory):
        torch.save(self.model.state_dict(), directory / 'detector.pt')
        (directory / 'adaptation.json').write_text(json.dumps(self.events, indent=2, allow_nan=False))
        if hasattr(self, 'reconstruction_threshold'):
            (directory / 'calibration.json').write_text(json.dumps({
                'reconstruction_threshold': self.reconstruction_threshold,
                'reference_distribution': self.reference_distribution.tolist()}))

    def statistics(self):
        return {'chunks': self.chunk_index, 'accepted_updates': self.accepted,
                'rejected_updates': self.rejected, 'advisor_fallbacks': self.fallbacks}
