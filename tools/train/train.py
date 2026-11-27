#!/usr/bin/env python3
"""Train, quantize, and export a `string2fingerprint` model.

The network and its integer arithmetic are specified in `docs/design.md` and
implemented in `src/model.rs`. This script trains a float network on
fingerprints, quantizes it to that arithmetic, evaluates the quantized
network with an exact integer simulation, and writes the model in the `S2FP`
format together with golden vectors that the Rust tests replay.

Only NumPy is required. See `README.md` in this directory for the procedure.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINGERPRINT_LEN = 3072
CATEGORIES = (
    "malformed",
    "obfuscation",
    "sql_injection",
    "markup_injection",
    "path_traversal",
    "command_injection",
    "template_injection",
    "header_injection",
)
FORMAT_VERSION = 1
MAX_SHIFT = 62
# The output layer stores logits in units of 1/16, clamped to -128..=127.
LOGIT_SCALE = 16.0
LOGISTIC = np.array(
    [round(255 / (1 + math.exp(-(index - 128) / LOGIT_SCALE))) for index in range(256)],
    dtype=np.uint8,
)


def load_split(prefix: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return features as `uint8[n, FINGERPRINT_LEN]` and labels as `float32[n, 8]`."""
    features = np.fromfile(f"{prefix}.x", dtype=np.uint8).reshape(-1, FINGERPRINT_LEN)
    masks = np.fromfile(f"{prefix}.y", dtype=np.uint8)
    if len(masks) != len(features):
        raise ValueError(f"{prefix}: feature and label counts differ")
    labels = ((masks[:, None] >> np.arange(len(CATEGORIES))) & 1).astype(np.float32)
    return features, labels


# --------------------------------------------------------------------------
# Float training
# --------------------------------------------------------------------------


@dataclass
class FloatModel:
    """Weights `[out, in]` and biases per layer, on activations in 0..1."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def forward(self, inputs: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Return the pre-activations of every layer and the logits."""
        pre_activations = []
        activation = inputs
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre = activation @ weight.T + bias
            pre_activations.append(pre)
            if index + 1 < len(self.weights):
                # The clamp to 0..1 is the rectifier and the saturation of
                # the unsigned 8-bit activation.
                activation = np.clip(pre, 0.0, 1.0)
        return pre_activations, pre_activations[-1]


def init_model(rng: np.random.Generator, dimensions: list[int]) -> FloatModel:
    weights, biases = [], []
    for fan_in, fan_out in zip(dimensions, dimensions[1:]):
        scale = math.sqrt(2.0 / fan_in)
        weights.append((rng.standard_normal((fan_out, fan_in)) * scale).astype(np.float32))
        biases.append(np.zeros(fan_out, dtype=np.float32))
    return FloatModel(weights, biases)


def fake_quantize_rows(weight: np.ndarray) -> np.ndarray:
    """Round each row to the 255 levels it will have after quantization."""
    scale = np.maximum(np.abs(weight).max(axis=1, keepdims=True), 1e-12) / 127.0
    return np.round(weight / scale) * scale


def train(
    rng: np.random.Generator,
    features: np.ndarray,
    labels: np.ndarray,
    dimensions: list[int],
    epochs: int,
    quantized_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    input_dropout: float,
    log,
    benign_weight: float = 1.0,
) -> FloatModel:
    """Train with Adam on binary cross-entropy, one sigmoid per category.

    During the last `quantized_epochs` epochs the forward pass uses weights
    rounded to their 8-bit levels and activations rounded to 1/255, with
    straight-through gradients, so the network adapts to quantization.

    `benign_weight` scales the loss of samples without any label. A value
    above 1 trades recall for fewer false positives on benign text.
    """
    model = init_model(rng, dimensions)
    parameters = model.weights + model.biases
    first = [np.zeros_like(parameter) for parameter in parameters]
    second = [np.zeros_like(parameter) for parameter in parameters]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    steps_per_epoch = len(features) // batch_size
    total_steps = epochs * steps_per_epoch
    step = 0
    for epoch in range(epochs):
        quantized = epoch >= epochs - quantized_epochs
        order = rng.permutation(len(features))
        loss_sum = 0.0
        for batch in range(steps_per_epoch):
            rows = order[batch * batch_size : (batch + 1) * batch_size]
            inputs = features[rows].astype(np.float32) / 255.0
            targets = labels[rows]
            if input_dropout > 0:
                keep = rng.random(inputs.shape, dtype=np.float32) >= input_dropout
                inputs = inputs * keep

            forward_weights = (
                [fake_quantize_rows(weight) for weight in model.weights]
                if quantized
                else model.weights
            )
            activations = [inputs]
            pre_activations = []
            for index, (weight, bias) in enumerate(zip(forward_weights, model.biases)):
                pre = activations[-1] @ weight.T + bias
                pre_activations.append(pre)
                if index + 1 < len(forward_weights):
                    activation = np.clip(pre, 0.0, 1.0)
                    if quantized:
                        activation = np.round(activation * 255.0) / 255.0
                    activations.append(activation)
            logits = pre_activations[-1]
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            loss_sum += float(
                np.mean(np.logaddexp(0.0, logits) - targets * logits)
            )

            gradient = (probabilities - targets) / len(rows)
            if benign_weight != 1.0:
                weights = np.where(targets.sum(axis=1, keepdims=True) == 0, benign_weight, 1.0).astype(np.float32)
                gradient = gradient * weights
            weight_gradients = [None] * len(model.weights)
            bias_gradients = [None] * len(model.weights)
            for index in range(len(model.weights) - 1, -1, -1):
                weight_gradients[index] = gradient.T @ activations[index]
                bias_gradients[index] = gradient.sum(axis=0)
                if index > 0:
                    gradient = gradient @ forward_weights[index]
                    pre = pre_activations[index - 1]
                    gradient = gradient * ((pre > 0.0) & (pre < 1.0))

            step += 1
            rate = learning_rate * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
            gradients = weight_gradients + bias_gradients
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                first[index] = beta1 * first[index] + (1 - beta1) * gradient
                second[index] = beta2 * second[index] + (1 - beta2) * gradient * gradient
                corrected_first = first[index] / (1 - beta1**step)
                corrected_second = second[index] / (1 - beta2**step)
                parameter -= rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
                if index < len(model.weights):
                    parameter -= rate * weight_decay * parameter
        log(
            f"epoch {epoch + 1}/{epochs}"
            f"{' (quantized)' if quantized else ''}: loss {loss_sum / steps_per_epoch:.5f}"
        )
    return model


# --------------------------------------------------------------------------
# Quantization and exact integer inference
# --------------------------------------------------------------------------


@dataclass
class QuantizedLayer:
    """One layer in the integer arithmetic of `src/model.rs`."""

    shift: int
    bias: np.ndarray  # int64[out], each fits in i32
    multiplier: np.ndarray  # int64[out], each in 0..2^31-1
    weights: np.ndarray  # int8[out, in], each in -127..=127


def quantize_layer(weight: np.ndarray, bias: np.ndarray, output_scale: float) -> QuantizedLayer:
    """Quantize `output = output_scale * (weight @ activation + bias)`.

    Activations are stored as `255 * activation`. Row `r` gets its own weight
    scale `s[r]`, so that `acc = sum(wq * 255 * a)` equals
    `255 * (w @ a) / s[r]`, and the stored output is
    `(acc + bq) * s[r] * output_scale / 255`.
    """
    weight = weight.astype(np.float64)
    bias = bias.astype(np.float64)
    row_max = np.abs(weight).max(axis=1)
    # A row with no usable weights keeps a scale that still represents its bias.
    scale = np.maximum(row_max / 127.0, np.abs(bias) * 255.0 / 2**30)
    scale = np.maximum(scale, 1e-9)
    quantized = np.clip(np.round(weight / scale[:, None]), -127, 127).astype(np.int8)
    quantized_bias = np.round(bias * 255.0 / scale).astype(np.int64)

    factor = scale * output_scale / 255.0
    shift = min(MAX_SHIFT, int(math.floor(30 - math.log2(factor.max()))))
    if shift < 1:
        raise ValueError("layer scale is too large to represent")
    multiplier = np.round(factor * 2.0**shift).astype(np.int64)
    layer = QuantizedLayer(shift, quantized_bias, multiplier, quantized)
    validate_layer(layer)
    return layer


def validate_layer(layer: QuantizedLayer) -> None:
    """Reject a layer that `Model::from_bytes` would reject.

    The accumulator bound here combines the widest row with the largest bias
    of any row. That is stricter than the per-row bound of the reader, so a
    layer that passes here always loads.
    """
    if not 1 <= layer.shift <= MAX_SHIFT:
        raise ValueError("shift out of range")
    if layer.multiplier.min() < 0 or layer.multiplier.max() > 2**31 - 1:
        raise ValueError("multiplier out of range")
    if layer.weights.min() < -127:
        raise ValueError("weight out of range")
    max_dot = np.abs(layer.weights.astype(np.int64)).sum(axis=1).max() * 255
    if max_dot + np.abs(layer.bias).max() > 2**31 - 1:
        raise ValueError("accumulator can overflow")


def quantize(model: FloatModel) -> list[QuantizedLayer]:
    layers = []
    for index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        last = index + 1 == len(model.weights)
        layers.append(quantize_layer(weight, bias, LOGIT_SCALE if last else 255.0))
    return layers


def infer(layers: list[QuantizedLayer], features: np.ndarray) -> np.ndarray:
    """Return `uint8[n, 8]` scores, bit-identical to the Rust inference."""
    activation = features.astype(np.int64)
    for index, layer in enumerate(layers):
        acc = activation @ layer.weights.astype(np.int64).T
        value = ((acc + layer.bias) * layer.multiplier + (1 << (layer.shift - 1))) >> layer.shift
        if index + 1 < len(layers):
            activation = np.clip(value, 0, 255)
        else:
            return LOGISTIC[np.clip(value, -128, 127) + 128]
    raise ValueError("model has no layers")


def encode(layers: list[QuantizedLayer]) -> bytes:
    """Serialize in the `S2FP` format documented in `src/model.rs`."""
    dimensions = [layers[0].weights.shape[1]] + [layer.weights.shape[0] for layer in layers]
    out = bytearray(b"S2FP")
    out += struct.pack("<HBB", FORMAT_VERSION, len(layers), 0)
    out += struct.pack(f"<{len(dimensions)}H", *dimensions)
    for layer in layers:
        out += struct.pack("<B3x", layer.shift)
        out += layer.bias.astype("<i4").tobytes()
        out += layer.multiplier.astype("<i4").tobytes()
        out += layer.weights.astype(np.int8).tobytes()
    checksum = 0x811C9DC5
    for byte in out:
        checksum = ((checksum ^ byte) * 0x01000193) & 0xFFFFFFFF
    out += struct.pack("<I", checksum)
    return bytes(out)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Area under the ROC curve, with ties counted as half."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    positive_count = positives.sum()
    negative_count = len(positives) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    rank_sum = ranks[positives.astype(bool)].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    )


def evaluate(scores: np.ndarray, labels: np.ndarray, thresholds: tuple[int, ...]) -> dict:
    benign = labels.sum(axis=1) == 0
    report = {"samples": int(len(labels)), "benign": int(benign.sum()), "categories": {}}
    for index, name in enumerate(CATEGORIES):
        positive = labels[:, index] == 1
        entry = {"positives": int(positive.sum()), "auc": round(auc(scores[:, index], positive), 5)}
        for threshold in thresholds:
            flagged = scores[:, index] >= threshold
            entry[f"recall@{threshold}"] = round(float(flagged[positive].mean()), 5)
            entry[f"benign_fpr@{threshold}"] = round(float(flagged[benign].mean()), 5)
            entry[f"other_attack_fpr@{threshold}"] = round(
                float(flagged[~positive & ~benign].mean()), 5
            )
        report["categories"][name] = entry
    for threshold in thresholds:
        any_flagged = (scores >= threshold).any(axis=1)
        report[f"benign_any_fpr@{threshold}"] = round(float(any_flagged[benign].mean()), 5)
        report[f"attack_any_recall@{threshold}"] = round(float(any_flagged[~benign].mean()), 5)
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="directory with train/val/test .x and .y files")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--hidden", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--quantized-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-dropout", type=float, default=0.1)
    parser.add_argument("--benign-weight", type=float, default=1.0, help="loss weight of benign samples")
    parser.add_argument("--golden", type=int, default=64, help="number of golden vectors")
    parser.add_argument("--name", default="default", help="base name of the output files")
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    rng = np.random.default_rng(arguments.seed)
    train_features, train_labels = load_split(arguments.data / "train")
    validation_features, validation_labels = load_split(arguments.data / "val")
    test_features, test_labels = load_split(arguments.data / "test")
    log(f"train {len(train_features)}, val {len(validation_features)}, test {len(test_features)}")

    dimensions = [FINGERPRINT_LEN, *arguments.hidden, len(CATEGORIES)]
    model = train(
        rng,
        train_features,
        train_labels,
        dimensions,
        arguments.epochs,
        arguments.quantized_epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.weight_decay,
        arguments.input_dropout,
        log,
        arguments.benign_weight,
    )

    layers = quantize(model)
    thresholds = (128, 192, 230)
    _, float_logits = model.forward(validation_features.astype(np.float32) / 255.0)
    float_scores = 255.0 / (1.0 + np.exp(-float_logits))
    validation_scores = infer(layers, validation_features)
    drift = np.abs(validation_scores.astype(np.float64) - float_scores)
    report = {
        "dimensions": dimensions,
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(arguments).items()
        },
        "quantization_drift": {
            "mean_abs_score": round(float(drift.mean()), 4),
            "max_abs_score": round(float(drift.max()), 2),
        },
        "validation": evaluate(validation_scores, validation_labels, thresholds),
        "test": evaluate(infer(layers, test_features), test_labels, thresholds),
    }

    arguments.out.mkdir(parents=True, exist_ok=True)
    encoded = encode(layers)
    (arguments.out / f"{arguments.name}.s2fp").write_bytes(encoded)
    golden_rows = rng.choice(len(test_features), size=arguments.golden, replace=False)
    golden = bytearray()
    for row in sorted(golden_rows):
        golden += test_features[row].tobytes()
        golden += infer(layers, test_features[row : row + 1])[0].tobytes()
    (arguments.out / f"{arguments.name}.golden").write_bytes(golden)
    (arguments.out / f"{arguments.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {len(encoded)} model bytes, {arguments.golden} golden vectors")
    print(json.dumps(report["test"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
