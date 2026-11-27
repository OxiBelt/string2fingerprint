"""Tests for the quantization and export code of `train.py`.

They need NumPy and are skipped without it.
"""

from __future__ import annotations

import struct
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    import train


@unittest.skipIf(np is None, "NumPy is not installed")
class QuantizationTest(unittest.TestCase):
    def float_model(self, seed: int, dimensions: list[int]) -> "train.FloatModel":
        rng = np.random.default_rng(seed)
        model = train.init_model(rng, dimensions)
        for bias in model.biases:
            bias += rng.standard_normal(bias.shape).astype(np.float32) * 0.1
        return model

    def test_quantized_scores_track_the_float_model(self) -> None:
        dimensions = [train.FINGERPRINT_LEN, 16, 12, len(train.CATEGORIES)]
        model = self.float_model(3, dimensions)
        rng = np.random.default_rng(4)
        # Sparse inputs with capped values, like real fingerprints.
        features = np.where(
            rng.random((64, train.FINGERPRINT_LEN)) < 0.05,
            rng.integers(0, 256, (64, train.FINGERPRINT_LEN)),
            0,
        ).astype(np.uint8)
        _, logits = model.forward(features.astype(np.float32) / 255.0)
        expected = 255.0 / (1.0 + np.exp(-logits))
        scores = train.infer(train.quantize(model), features)
        self.assertEqual(scores.dtype, np.uint8)
        self.assertEqual(scores.shape, (64, len(train.CATEGORIES)))
        self.assertLess(np.abs(scores - expected).mean(), 4.0)

    def test_layers_pass_the_checks_of_the_reader(self) -> None:
        model = self.float_model(5, [train.FINGERPRINT_LEN, 8, len(train.CATEGORIES)])
        # A row of zero weights keeps its bias.
        model.weights[0][0] = 0.0
        model.biases[0][0] = 0.5
        for layer in train.quantize(model):
            train.validate_layer(layer)
            self.assertGreaterEqual(int(layer.weights.min()), -127)
            self.assertLess(int(np.abs(layer.bias).max()), 2**31)

    def test_encoding_has_the_documented_layout(self) -> None:
        dimensions = [train.FINGERPRINT_LEN, 8, len(train.CATEGORIES)]
        encoded = train.encode(train.quantize(self.float_model(6, dimensions)))
        self.assertEqual(encoded[:4], b"S2FP")
        version, layer_count, reserved = struct.unpack_from("<HBB", encoded, 4)
        self.assertEqual((version, layer_count, reserved), (train.FORMAT_VERSION, 2, 0))
        self.assertEqual(list(struct.unpack_from("<3H", encoded, 8)), dimensions)
        size = 14
        for inputs, outputs in zip(dimensions, dimensions[1:]):
            size += 4 + 8 * outputs + inputs * outputs
        self.assertEqual(len(encoded), size + 4)
        checksum = 0x811C9DC5
        for byte in encoded[:-4]:
            checksum = ((checksum ^ byte) * 0x01000193) & 0xFFFFFFFF
        self.assertEqual(struct.unpack_from("<I", encoded, size)[0], checksum)

    def test_benign_weight_lowers_scores_of_benign_samples(self) -> None:
        rng = np.random.default_rng(7)
        features = np.where(
            rng.random((768, train.FINGERPRINT_LEN)) < 0.03,
            rng.integers(0, 256, (768, train.FINGERPRINT_LEN)),
            0,
        ).astype(np.uint8)
        labels = np.zeros((768, len(train.CATEGORIES)), dtype=np.float32)
        # Half of the samples carry a label that depends weakly on the input.
        labels[: 384, 2] = (features[: 384, :64].sum(axis=1) > np.median(features[: 384, :64].sum(axis=1))).astype(np.float32)
        labels[: 384, 2] = np.maximum(labels[: 384, 2], rng.random(384) < 0.5)
        benign = labels.sum(axis=1) == 0
        means = []
        for weight in (1.0, 8.0):
            model = train.train(np.random.default_rng(1), features, labels, [train.FINGERPRINT_LEN, 8, len(train.CATEGORIES)],
                                epochs=3, quantized_epochs=1, batch_size=64, learning_rate=2e-3, weight_decay=0.0,
                                input_dropout=0.0, log=lambda _: None, benign_weight=weight)
            means.append(float(train.infer(train.quantize(model), features[benign])[:, 2].mean()))
        self.assertLess(means[1], means[0])

    def test_auc_counts_ties_as_half(self) -> None:
        scores = np.array([0, 0, 1, 1])
        self.assertEqual(train.auc(scores, np.array([0, 1, 0, 1])), 0.5)
        self.assertEqual(train.auc(scores, np.array([0, 0, 1, 1])), 1.0)


if __name__ == "__main__":
    unittest.main()
