import unittest

import numpy as np

from nn_xray_lab.core import BinaryMLP, binary_cross_entropy, gradient_check
from nn_xray_lab.datasets import generate_dataset, split_and_scale
from nn_xray_lab.evaluation import evaluate


class DatasetTests(unittest.TestCase):
    def test_all_datasets_are_reproducible_binary_2d(self):
        for name in ("XOR", "Two Moons", "Concentric Circles", "Gaussian Blobs", "Spiral"):
            x1, y1 = generate_dataset(name, samples=500, seed=12)
            x2, y2 = generate_dataset(name, samples=500, seed=12)
            self.assertEqual(x1.shape, (500, 2))
            self.assertEqual(y1.shape, (500,))
            self.assertTrue(np.array_equal(x1, x2))
            self.assertTrue(np.array_equal(y1, y2))
            self.assertEqual(set(y1), {0, 1})

    def test_split_is_disjoint_and_scaler_fits_training_only(self):
        x, y = generate_dataset("Two Moons", samples=500, seed=4)
        split = split_and_scale(x, y, seed=4)
        self.assertEqual(len(split.y_train) + len(split.y_validation) + len(split.y_test), 500)
        self.assertTrue(np.allclose(split.x_train.mean(axis=0), 0, atol=1e-12))
        self.assertFalse(np.allclose(split.x_test.mean(axis=0), 0, atol=1e-12))
        self.assertEqual(len(split.y_train), 300)
        self.assertEqual(len(split.y_validation), 100)
        self.assertEqual(len(split.y_test), 100)


class ModelTests(unittest.TestCase):
    def test_forward_probability_shape_and_range(self):
        model = BinaryMLP((3, 2), "relu", seed=1)
        probabilities = model.predict_proba(np.zeros((7, 2)))
        self.assertEqual(probabilities.shape, (7, 1))
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))

    def test_loss_is_finite_at_extremes(self):
        value = binary_cross_entropy(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]))
        self.assertTrue(np.isfinite(value))

    def test_gradient_check(self):
        result = gradient_check()
        self.assertTrue(result["passed"], result)

    def test_training_changes_parameters_and_records_matching_history(self):
        x, y = generate_dataset("XOR", samples=500, noise=.05, seed=2)
        split = split_and_scale(x, y, seed=2)
        model = BinaryMLP((8, 8), "tanh", learning_rate=.1, seed=2)
        history = model.fit(split.x_train, split.y_train, split.x_validation, split.y_validation, epochs=50, batch_size=32, record_every=5)
        self.assertGreater(np.linalg.norm(model.weights[0] - model.initial_weights[0]), 0)
        self.assertEqual(len(history.epochs), len(history.train_loss))
        self.assertEqual(len(history.epochs), len(history.validation_accuracy))
        self.assertEqual(model.parameter_count, sum(w.size + b.size for w, b in zip(model.weights, model.biases)))

    def test_evaluation_metrics_have_valid_shapes_and_ranges(self):
        x, y = generate_dataset("Gaussian Blobs", samples=500, separation=3, noise=.1, seed=9)
        split = split_and_scale(x, y, seed=9)
        model = BinaryMLP((4,), "tanh", learning_rate=.1, seed=9)
        model.fit(split.x_train, split.y_train, split.x_validation, split.y_validation, epochs=10, record_every=5)
        report = evaluate(model, split.x_test, split.y_test)
        self.assertEqual(report["confusion_matrix"].shape, (2, 2))
        self.assertEqual(report["probabilities"].shape, (len(split.y_test),))
        self.assertTrue(np.all((report["probabilities"] > 0) & (report["probabilities"] < 1)))
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertGreaterEqual(report[key], 0.0)
            self.assertLessEqual(report[key], 1.0)


if __name__ == "__main__":
    unittest.main()
