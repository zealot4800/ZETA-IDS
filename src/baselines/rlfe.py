"""FE-Net prototype training and causal predicted-label retrieval updates."""
from __future__ import annotations

import json
import sys
import numpy as np
import torch
from torch import nn
from scipy.linalg import pinvh

from .common import VectorLibrary


class FENet(nn.Module):
    def __init__(self):
        super().__init__()
        # Fig. 3: one channel, 128 positions; pooling gives 64 x 64 = 4096.
        channels = [1, 16, 32, 64, 64]
        self.convolutions = nn.Sequential(*[
            block for a, b in zip(channels, channels[1:])
            for block in (nn.Conv1d(a, b, kernel_size=1), nn.BatchNorm1d(b), nn.ReLU())])
        self.pool = nn.AvgPool1d(2)
        self.projection = nn.Sequential(nn.Linear(4096, 2048), nn.Linear(2048, 1024))

    def forward(self, values):
        if values.shape[1] < 128:
            values = nn.functional.pad(values, (0, 128 - values.shape[1]))
        values = values[:, :128].unsqueeze(1)
        return self.projection(self.pool(self.convolutions(values)).flatten(1))


class RLFEIDS:
    def __init__(self, reasoner, options, values, raw, labels, seed):
        torch.manual_seed(seed)
        self.reasoner, self.options = reasoner, options
        requested = options.get('device', 'auto')
        if requested not in ('auto', 'cpu', 'cuda'):
            raise ValueError('rlfe_ids.device must be auto, cpu, or cuda')
        if requested == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('RLFE-IDS requires CUDA, but this Python environment cannot use it')
        self.device = torch.device('cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu')
        # Keep the full cohort on CPU; transfer only bounded batches to the GPU.
        self.values = torch.as_tensor(values)
        self.raw, self.labels = raw, list(labels)
        self.classes = sorted(set(labels))
        self.targets = torch.tensor([self.classes.index(label) for label in labels])
        self.model, self.library = FENet().to(self.device), VectorLibrary()
        print(json.dumps({'event': 'compute_device', 'method': 'rlfe_ids',
                          'device': str(self.device),
                          'name': torch.cuda.get_device_name(self.device)
                          if self.device.type == 'cuda' else 'CPU'}), file=sys.stderr, flush=True)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.precision = None

    def encode(self, values):
        self.model.eval()
        with torch.no_grad():
            return torch.cat([self.model(batch.to(self.device)).cpu() for batch in
                              torch.as_tensor(values).split(self.options['batch_size'])]).numpy()

    def fit(self, directory):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.options['learning_rate'])
        for epoch in range(self.options['epochs']):
            self.model.train()
            # Refresh class prototypes each epoch; detach the support path.
            with torch.no_grad():
                prototypes = []
                for label in range(len(self.classes)):
                    indices = np.flatnonzero(self.targets.numpy() == label)
                    chosen = self.rng.permutation(indices)[:self.options['prototypes_per_class']]
                    prototypes.append(self.model(self.values[chosen].to(self.device)).mean(0))
                prototypes = torch.stack(prototypes)
            order = self.rng.permutation(len(self.values))
            for start in range(0, len(order), self.options['batch_size']):
                indices = order[start:start + self.options['batch_size']]
                embedding = self.model(self.values[indices].to(self.device))
                logits = -torch.cdist(embedding, prototypes)
                loss = nn.functional.cross_entropy(logits, self.targets[indices].to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(json.dumps({'event': 'training_progress', 'method': 'rlfe_ids',
                              'epoch': epoch + 1, 'epochs': self.options['epochs'],
                              'last_batch_loss': float(loss.detach())}), file=sys.stderr, flush=True)
        embedded = self.encode(self.values.numpy())
        if self.options['distance'] == 'mahalanobis':
            covariance = np.cov(embedded, rowvar=False) if len(embedded) > 1 else np.eye(1024)
            self.precision = pinvh(covariance + np.eye(1024) * 1e-3)
        selected = [int(i) for label in self.classes for i in
                    self.rng.permutation(np.flatnonzero(np.asarray(self.labels) == label))
                    [:self.options['references_per_class']]]
        for i in selected:
            self.library.add(embedded[i], {'flow': self.raw[i], 'label': self.labels[i],
                                          'provenance': 'bootstrap_training'})

    def predict(self, flow, vector):
        embedding = self.encode(np.asarray(vector, dtype=np.float32)[None])[0]
        context = [{'reference': entry, 'distance': -score} for entry, score in
                   self.library.nearest(embedding, self.options['retrieval_k'],
                                        cosine=False, precision=self.precision)]
        result = self.reasoner.classify(flow, context)
        # Append only after obtaining the prediction; never attach replay truth.
        self.library.add(embedding, {'flow': flow, 'label': result['label'],
                                     'provenance': 'replay_prediction'})
        self.updates += 1
        return {**result, 'retrieved_examples': len(context)}

    def save(self, directory):
        if directory is None:
            return
        self.library.save(directory)
        torch.save({name: value.cpu() for name, value in self.model.state_dict().items()},
                   directory / 'fe_net.pt')
        if self.precision is not None:
            np.save(directory / 'mahalanobis_precision.npy', self.precision)

    def statistics(self):
        return {'reference_entries': len(self.library.entries), 'predicted_label_updates': self.updates,
                'device': str(self.device)}
