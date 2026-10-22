"""Paper method: predicted-class SHAP reconstruction + epistemic uncertainty + GMM.

Fawaz et al., DOI 10.1109/TNSM.2026.3731401, Sections IV-C/D.
Author settings and documented adaptations are in config/loao.yaml.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import entr, softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .loao_data import _partitions
from .loao_model import _preprocessor


def validate_config(cfg: dict) -> None:
    p = cfg["paper"]
    if p["classifier"]["backend"] not in ("rf", "xgboost"):
        raise ValueError("Paper classifier must be rf or xgboost")
    if not 0 < p["shap"]["importance_fraction"] <= 1 or not 0 < p["threshold"]["quantile"] < 1:
        raise ValueError("Invalid SHAP coverage or novelty quantile")
    ae, resource = p["reconstruction"], p["resources"]
    positive = [resource["cpu_threads"], resource["tensorflow_inter_threads"], resource["batch_size"],
                ae["batch_size"], ae["max_epochs"], ae["patience"], p["uncertainty"]["subensembles"],
                *ae["encoder_units"], *ae["decoder_units"], *p["gmm"]["components"]]
    if any(type(value) is not int or value < 1 for value in positive):
        raise ValueError("Paper counts and layer widths must be positive integers")
    if not ae["encoder_units"] or ae["decoder_units"] != list(reversed(ae["encoder_units"])):
        raise ValueError("The paper requires a symmetric encoder/decoder")
    if ae["activation"] != "relu" or ae["output_activation"] != "tanh" or ae["loss"] != "mse":
        raise ValueError("The paper reconstruction uses ReLU, tanh and MSE")
    if not 0 <= ae["dropout"] < 1 or not 0 < ae["train_fraction"] < 1:
        raise ValueError("Invalid reconstruction dropout or training fraction")
    if p["shap"]["model_output"] != "raw" or p["shap"]["feature_perturbation"] != "tree_path_dependent":
        raise ValueError("The supported paper SHAP representation is tree-path-dependent raw output")
    if resource["dtype"] not in ("float32", "float64") or not np.isfinite(resource["max_dense_mb"]) or resource["max_dense_mb"] <= 0:
        raise ValueError("Invalid paper memory settings")
    pre = p["preprocessing"]
    if not 0 < pre["correlation_threshold"] <= 1 or pre["variance_threshold"] < 0 or pre["skew_threshold"] < 0:
        raise ValueError("Invalid paper feature filtering settings")
    if len(pre["winsor_limits"]) != 2 or any(not 0 <= q < .5 for q in pre["winsor_limits"]):
        raise ValueError("Invalid winsorization tails")
    if not p["gmm"]["covariance_types"] or not set(p["gmm"]["covariance_types"]) <= {"full", "diag", "tied", "spherical"}:
        raise ValueError("Invalid GMM covariance types")
    limit = p["shap"]["selection_max_rows"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("SHAP selection_max_rows must be null or positive")


def _memory(shape: tuple, options: dict, itemsize: int = np.dtype(float).itemsize) -> None:
    if np.prod(shape, dtype=np.int64) * itemsize > options["resources"]["max_dense_mb"] * 1024**2:
        raise MemoryError("Paper array exceeds max_dense_mb; reduce batching or explicit comparison row limits")


def _dense(values, options: dict) -> np.ndarray:
    _memory(values.shape, options, max(values.dtype.itemsize, np.dtype(options["resources"]["dtype"]).itemsize))
    return (values.toarray() if hasattr(values, "toarray") else values).astype(options["resources"]["dtype"], copy=False)


def _fit_features(data: dict, rows: np.ndarray, p: dict) -> tuple[dict, np.ndarray]:
    pre = _preprocessor(data["numeric"], data["categorical"])
    values = _dense(pre.fit_transform(data["x"].iloc[rows]), p)
    names = np.asarray(pre.get_feature_names_out())
    options = p["preprocessing"]
    variance = np.flatnonzero(np.var(values, axis=0, dtype=np.float64) > options["variance_threshold"])
    if not variance.size:
        raise ValueError("No source-training features survive the paper variance filter")
    values, names = values[:, variance], names[variance]
    skewed = np.flatnonzero(np.abs(pd.DataFrame(values).skew().to_numpy()) > options["skew_threshold"])
    ordered = np.sort(values[:, skewed], axis=0)
    low, high = options["winsor_limits"]
    lower = ordered[int(len(values) * low)]
    upper = ordered[len(values) - int(len(values) * high) - 1]
    values[:, skewed] = np.clip(values[:, skewed], lower, upper)
    _memory((values.shape[1], values.shape[1]), p)
    corr = pd.DataFrame(values).corr().abs().to_numpy()
    keep = np.flatnonzero(~np.any(np.triu(corr > options["correlation_threshold"], k=1), axis=0))
    scaler = StandardScaler() if options["standardize"] else None
    transformed = scaler.fit_transform(values[:, keep]) if scaler else values[:, keep]
    return {"preprocessor": pre, "variance": variance, "skewed": skewed, "lower": lower,
            "upper": upper, "keep": keep, "scaler": scaler, "feature_names": names[keep].tolist()}, transformed


def _transform(model: dict, frame: pd.DataFrame) -> np.ndarray:
    pre = model["features"]
    values = _dense(pre["preprocessor"].transform(frame), model["options"])[:, pre["variance"]]
    values[:, pre["skewed"]] = np.clip(values[:, pre["skewed"]], pre["lower"], pre["upper"])
    values = values[:, pre["keep"]]
    return pre["scaler"].transform(values) if pre["scaler"] else values


def _shap(model: dict, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, classifier = model["options"], model["classifier"]
    predictions = np.argmax(classifier.predict_proba(values), axis=1)
    _memory((len(values), values.shape[1], len(model["classes"])), p)
    raw = model["explainer"].shap_values(values,
        approximate=p["shap"]["approximate"][p["classifier"]["backend"]],
        check_additivity=p["shap"]["check_additivity"])
    phi = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)
    expected = (len(values), values.shape[1], len(model["classes"]))
    if phi.shape != expected:
        raise ValueError(f"Unexpected TreeSHAP shape {phi.shape}; expected {expected}")
    return phi[np.arange(len(values)), :, predictions], predictions


def _epistemic(model: dict, values: np.ndarray) -> np.ndarray:
    classifier, p = model["classifier"], model["options"]
    rf = p["classifier"]["backend"] == "rf"
    size = len(classifier.estimators_) if rf else classifier.get_booster().num_boosted_rounds()
    count = p["uncertainty"]["subensembles"]
    if size < count:
        raise ValueError("The classifier has fewer trees/boosting rounds than uncertainty subensembles")
    mean = np.zeros((len(values), len(model["classes"])))
    mean_entropy = np.zeros(len(values))
    for group in np.array_split(np.arange(size), count):
        if rf:
            probabilities = sum(classifier.estimators_[i].predict_proba(values) for i in group) / len(group)
        else:
            margins = classifier.predict(values, output_margin=True,
                iteration_range=(int(group[0]), int(group[-1]) + 1))
            probabilities = softmax(margins.reshape(len(values), -1), axis=1)
        mean += probabilities / count
        mean_entropy += entr(probabilities).sum(axis=1) / count
    # Eqs. (2)-(4), natural-log entropy. RF probabilities follow the author code.
    return np.maximum(entr(mean).sum(axis=1) - mean_entropy, 0)


def _autoencoder(values: np.ndarray, inner_train: np.ndarray, p: dict) -> tuple[object, dict]:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = p["resources"]["tensorflow_log_level"]
    import tensorflow as tf
    ae, limits = p["reconstruction"], p["resources"]
    for get, set_, count in (
            (tf.config.threading.get_intra_op_parallelism_threads, tf.config.threading.set_intra_op_parallelism_threads, limits["cpu_threads"]),
            (tf.config.threading.get_inter_op_parallelism_threads, tf.config.threading.set_inter_op_parallelism_threads, limits["tensorflow_inter_threads"])):
        if get() != count:
            set_(count)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(p["seed"])
    tf.config.experimental.enable_op_determinism()
    layers = [tf.keras.layers.Input(shape=(values.shape[1],))]
    for width in ae["encoder_units"] + ae["decoder_units"]:
        layers.extend([tf.keras.layers.Dense(width, kernel_initializer=ae["kernel_initializer"]),
                       tf.keras.layers.BatchNormalization(momentum=ae["batch_norm_momentum"], epsilon=ae["batch_norm_epsilon"]),
                       tf.keras.layers.Activation(ae["activation"]), tf.keras.layers.Dropout(ae["dropout"])])
    layers.append(tf.keras.layers.Dense(values.shape[1], activation=ae["output_activation"], kernel_initializer=ae["kernel_initializer"]))
    network = tf.keras.Sequential(layers)
    network.compile(optimizer=tf.keras.optimizers.Adam(**ae["optimizer"]), loss=ae["loss"])
    options = tf.data.Options()
    options.threading.private_threadpool_size = limits["cpu_threads"]
    options.threading.max_intra_op_parallelism = limits["cpu_threads"]
    datasets = []
    for mask, training in ((inner_train, True), (~inner_train, False)):
        subset = values[mask]
        if not len(subset):
            raise ValueError("Paper autoencoder requires nonempty source-training and early-stopping groups")
        dataset = tf.data.Dataset.from_tensor_slices((subset, subset))
        if training and ae["shuffle"]:
            dataset = dataset.shuffle(len(subset), seed=p["seed"])
        datasets.append(dataset.batch(ae["batch_size"]).with_options(options))
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ae["patience"],
                                            min_delta=ae["min_delta"], restore_best_weights=True)
    history = network.fit(datasets[0], validation_data=datasets[1], epochs=ae["max_epochs"],
                          callbacks=[early], shuffle=False, verbose=0)
    losses = history.history["val_loss"]
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite autoencoder validation loss")
    return network, {"epochs": len(losses), "best_validation_mse": float(min(losses)),
                     "inner_train_rows": int(inner_train.sum()), "inner_validation_rows": int((~inner_train).sum())}


def _reconstruction(model: dict, phi: np.ndarray) -> np.ndarray:
    reconstructed = model["autoencoder"](phi, training=False).numpy()
    return np.mean(np.square(phi - reconstructed), axis=1)


def fit(data: dict, train: np.ndarray, validation: np.ndarray, excluded: str, cfg: dict) -> dict:
    """Fit only on permitted source rows; evaluation frames and labels are never read."""
    validate_config(cfg)
    p = cfg["paper"]
    if np.any(train & ~data["train"]) or np.any(validation & ~data["validation"]):
        raise ValueError("Paper source rows must respect the shared partitions")
    rows = np.flatnonzero(train & (data["y"] == 1) & (data["family"] != excluded))
    val_rows = np.flatnonzero(validation & (data["y"] == 1) & (data["family"] != excluded))
    classes, labels = np.unique(data["family"][rows], return_inverse=True)
    if len(classes) < 2 or len(val_rows) < 2 or not set(data["family"][val_rows]) <= set(classes):
        raise ValueError("Paper fitting requires at least two known training families and supported validation attacks")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = p["resources"]["tensorflow_log_level"]
    import shap
    with threadpool_limits(limits=p["resources"]["cpu_threads"]):
        features, values = _fit_features(data, rows, p)
        backend = p["classifier"]["backend"]
        if backend == "rf":
            classifier = RandomForestClassifier(**p["classifier"]["rf"], random_state=p["seed"], n_jobs=p["resources"]["cpu_threads"])
        else:
            from xgboost import XGBClassifier
            classifier = XGBClassifier(**p["classifier"]["xgboost"], num_class=len(classes),
                                       random_state=p["seed"], n_jobs=p["resources"]["cpu_threads"])
        classifier.fit(values, labels)
        model = {"features": features, "classifier": classifier, "classes": classes, "options": p}
        model["explainer"] = shap.TreeExplainer(classifier, feature_perturbation=p["shap"]["feature_perturbation"], model_output=p["shap"]["model_output"])
        _memory(values.shape, p)
        phi = np.empty(values.shape, dtype=p["resources"]["dtype"])
        batch = p["resources"]["batch_size"]
        for start in range(0, len(values), batch):
            phi[start:start + batch] = _shap(model, values[start:start + batch])[0]
        limit = p["shap"]["selection_max_rows"]
        selection = np.arange(len(phi)) if limit is None else np.random.default_rng(p["seed"]).permutation(len(phi))[:limit]
        importance = np.mean(np.abs(phi[selection]), axis=0, dtype=np.float64)
        if not np.isfinite(importance).all() or importance.sum() <= 0:
            raise ValueError("Training SHAP influence is zero or nonfinite")
        order = np.argsort(-importance, kind="stable")
        count = np.searchsorted(np.cumsum(importance[order]), importance.sum() * p["shap"]["importance_fraction"]) + 1
        model["selected"] = order[:count]
        model["shap_scaler"] = StandardScaler()
        phi = model["shap_scaler"].fit_transform(phi[:, model["selected"]])
        ae = p["reconstruction"]
        _, partitions = _partitions(data["x"].iloc[rows], p["seed"] ^ ae["split_salt"],
                                    {"train": ae["train_fraction"], "validation": 1 - ae["train_fraction"]}, cfg["partition_buckets"])
        model["autoencoder"], ae_info = _autoencoder(phi, partitions == 0, p)
        pairs = np.empty((len(val_rows), 2))
        for start in range(0, len(val_rows), batch):
            transformed = _transform(model, data["x"].iloc[val_rows[start:start + batch]])
            selected_phi = _shap(model, transformed)[0][:, model["selected"]]
            scaled_phi = model["shap_scaler"].transform(selected_phi)
            pairs[start:start + batch, 0] = _reconstruction(model, scaled_phi)
            pairs[start:start + batch, 1] = _epistemic(model, transformed)
        model["joint_scaler"] = StandardScaler().fit(pairs)
        pairs = model["joint_scaler"].transform(pairs)
        options = p["gmm"]
        fits = []
        best, best_bic = None, np.inf
        for covariance in options["covariance_types"]:
            for components in options["components"]:
                if components > len(pairs):
                    continue
                candidate = GaussianMixture(n_components=components, covariance_type=covariance,
                    random_state=p["seed"], **{k: v for k, v in options.items() if k not in ("components", "covariance_types")})
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        candidate.fit(pairs)
                    bic = float(candidate.bic(pairs))
                except ValueError:
                    fits.append({"components": components, "covariance": covariance, "status": "numerically_invalid"})
                    continue
                fits.append({"components": components, "covariance": covariance, "bic": bic, "converged": bool(candidate.converged_)})
                if candidate.converged_ and np.isfinite(bic) and bic < best_bic:
                    best, best_bic = candidate, bic
        if best is None:
            raise ValueError("No finite converged source-validation GMM candidate")
        model["gmm"] = best
        model["threshold"] = float(np.quantile(-best.score_samples(pairs), p["threshold"]["quantile"], method=p["threshold"]["quantile_method"]))
    model["metadata"] = {**p["reference"], "backend": backend, "train_attack_rows": len(rows),
        "validation_attack_rows": len(val_rows), "class_mapping": {label: i for i, label in enumerate(classes)},
        "classifier_feature_names": features["feature_names"], "shap_selection_rows": len(selection),
        "shap_selected_features": np.asarray(features["feature_names"])[model["selected"]].tolist(),
        "autoencoder": ae_info, "gmm_selection": fits, "gmm_components": best.n_components,
        "gmm_covariance": best.covariance_type, "gmm_bic": best_bic, "novelty_threshold": model["threshold"]}
    return model


def predict(model: dict, frame: pd.DataFrame) -> dict:
    """Return native decisions and continuous novelty scores without evaluation labels."""
    p = model["options"]
    scores, reconstruction, epistemic = (np.empty(len(frame)) for _ in range(3))
    families = np.empty(len(frame), dtype=object)
    batch = p["resources"]["batch_size"]
    with threadpool_limits(limits=p["resources"]["cpu_threads"]):
        for start in range(0, len(frame), batch):
            end = start + batch
            values = _transform(model, frame.iloc[start:end])
            phi, predicted = _shap(model, values)
            scaled_phi = model["shap_scaler"].transform(phi[:, model["selected"]])
            reconstruction[start:end] = _reconstruction(model, scaled_phi)
            epistemic[start:end] = _epistemic(model, values)
            pairs = np.column_stack((reconstruction[start:end], epistemic[start:end]))
            scores[start:end] = -model["gmm"].score_samples(model["joint_scaler"].transform(pairs))
            families[start:end] = model["classes"][predicted]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite paper novelty score")
    return {"decisions": np.where(scores > model["threshold"], "ZERO_DAY_CANDIDATE", "KNOWN_ATTACK"),
            "known_family": families, "score": scores, "reconstruction_error": reconstruction,
            "epistemic_uncertainty": epistemic}
