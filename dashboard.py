

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from .core import BinaryMLP, TrainingHistory, gradient_check
from .datasets import DataSplit, generate_dataset, split_and_scale
from .evaluation import evaluate


COLORS = {"train": "#2563eb", "validation": "#f59e0b", "test": "#8b5cf6"}


@dataclass
class Experiment:
    name: str
    config: dict
    data: DataSplit
    model: BinaryMLP
    history: TrainingHistory
    train_metrics: dict
    validation_metrics: dict
    test_metrics: dict


def _architecture(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(not 1 <= value <= 64 for value in values):
        raise ValueError("Architecture must be comma-separated layer widths from 1 to 64, e.g. 8, 8")
    return values


def _scatter_splits(ax, data: DataSplit, scaler=None, predicted=None) -> None:
    groups = [("train", data.x_train, data.y_train), ("validation", data.x_validation, data.y_validation), ("test", data.x_test, data.y_test)]
    for split, x, y in groups:
        raw = scaler.inverse_transform(x) if scaler is not None else x
        correct = np.ones(len(y), dtype=bool) if predicted is None else predicted[split] == y
        for label, marker in [(0, "o"), (1, "^")]:
            mask = (y == label) & correct
            ax.scatter(raw[mask, 0], raw[mask, 1], s=22, marker=marker, color=COLORS[split], alpha=.72,
                       label=f"{split} · class {label}" if split == "train" else None)
        if predicted is not None:
            wrong = ~correct
            ax.scatter(raw[wrong, 0], raw[wrong, 1], s=55, marker="x", color="#dc2626", linewidth=1.8,
                       label="misclassified" if split == "train" else None)


def _dataset_plot(data: DataSplit, dataset_name: str, seed: int):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _scatter_splits(ax, data)
    counts = np.bincount(data.raw_y, minlength=2)
    ax.set(title=f"{dataset_name} — generated dataset", xlabel="Feature 1", ylabel="Feature 2")
    ax.legend(loc="best", ncols=2, fontsize=8)
    ax.text(.02, .02, f"n={len(data.raw_y)}  |  class 0={counts[0]}  |  class 1={counts[1]}\n"
                        f"split: train={len(data.y_train)}, validation={len(data.y_validation)}, test={len(data.y_test)}  |  seed={seed}",
            transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "#d1d5db"})
    fig.tight_layout()
    return fig


def _boundary_plot(experiment: Experiment):
    data, model = experiment.data, experiment.model
    raw = data.raw_x
    pad = .12 * max(np.ptp(raw[:, 0]), np.ptp(raw[:, 1]), 1.0)
    x_min, x_max = raw[:, 0].min() - pad, raw[:, 0].max() + pad
    y_min, y_max = raw[:, 1].min() - pad, raw[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    grid_raw = np.column_stack((gx.ravel(), gy.ravel()))
    probabilities = model.predict_proba(data.scaler.transform(grid_raw)).reshape(gx.shape)
    predicted = {name: model.predict(x) for name, x in [("train", data.x_train), ("validation", data.x_validation), ("test", data.x_test)]}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    surface = ax.contourf(gx, gy, probabilities, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.62, vmin=0, vmax=1)
    ax.contour(gx, gy, probabilities, levels=[.5], colors="#111827", linewidths=1.6)
    _scatter_splits(ax, data, data.scaler, predicted)
    fig.colorbar(surface, ax=ax, label="P(class 1)")
    ax.set(title="Decision boundary (black = P(class 1) = 0.5)", xlabel="Feature 1", ylabel="Feature 2")
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


def _training_plot(history: TrainingHistory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(history.epochs, history.train_loss, label="training", color=COLORS["train"])
    axes[0].plot(history.epochs, history.validation_loss, label="validation", color=COLORS["validation"])
    axes[0].set(title="Binary cross-entropy", xlabel="Actual epoch", ylabel="loss")
    axes[1].plot(history.epochs, np.array(history.train_accuracy) * 100, label="training", color=COLORS["train"])
    axes[1].plot(history.epochs, np.array(history.validation_accuracy) * 100, label="validation", color=COLORS["validation"])
    axes[1].set(title="Accuracy", xlabel="Actual epoch", ylabel="percent", ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    return fig


def _internals_plot(experiment: Experiment, neuron: int):
    model, data = experiment.model, experiment.data
    raw = data.raw_x
    gx, gy = np.meshgrid(np.linspace(raw[:, 0].min()-.2, raw[:, 0].max()+.2, 150), np.linspace(raw[:, 1].min()-.2, raw[:, 1].max()+.2, 150))
    grid = data.scaler.transform(np.column_stack((gx.ravel(), gy.ravel())))
    _, acts = model.forward(grid)
    first_hidden = acts[1]
    neuron = min(max(neuron, 0), first_hidden.shape[1] - 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # Activation map
    image = axes[0, 0].contourf(gx, gy, first_hidden[:, neuron].reshape(gx.shape), levels=20, cmap="viridis", vmin=0, vmax=1)
    _scatter_splits(axes[0, 0], data, data.scaler)
    fig.colorbar(image, ax=axes[0, 0], label=f"activation of neuron {neuron + 1}")
    axes[0, 0].set(title="First hidden-layer activation map", xlabel="Feature 1", ylabel="Feature 2")
    # Architecture diagram
    sizes = [2, *model.hidden_layers, 1]
    max_width = max(sizes)
    for layer, width in enumerate(sizes):
        ys = np.linspace(.12, .88, width)
        xs = np.full(width, layer)
        axes[0, 1].scatter(xs, ys, s=260, c="#dbeafe" if layer < len(sizes)-1 else "#fecaca", edgecolors="#1f2937", zorder=2)
        for source_y in ys:
            if layer < len(sizes)-1:
                next_y = np.linspace(.12, .88, sizes[layer+1])
                weights = model.weights[layer]
                source = list(ys).index(source_y)
                for target, target_y in enumerate(next_y):
                    value = weights[source, target]
                    axes[0, 1].plot([layer, layer+1], [source_y, target_y], color="#dc2626" if value < 0 else "#2563eb", alpha=min(.8, .12 + abs(value)/3), linewidth=.5 + min(2, abs(value)))
    axes[0, 1].set(title="Architecture / connection sign", xlim=(-.35, len(sizes)-.65), ylim=(0, 1)); axes[0, 1].axis("off")
    # Weight change / distribution
    changes = np.concatenate([(w - initial).ravel() for w, initial in zip(model.weights, model.initial_weights)])
    axes[1, 0].hist(changes, bins=25, color="#0f766e", alpha=.8)
    axes[1, 0].axvline(0, color="#111827", linewidth=1)
    axes[1, 0].set(title="Weight change from initialization", xlabel="final − initial weight", ylabel="count")
    # Numeric weights (first layer) + biases
    # Each column is one hidden neuron: two input weights plus its bias.
    matrix = np.vstack((model.weights[0], model.biases[0]))
    im = axes[1, 1].imshow(matrix, aspect="auto", cmap="coolwarm")
    axes[1, 1].set(title="First-layer weights and biases", xlabel="hidden neuron", ylabel="parameter")
    axes[1, 1].set_yticks([0, 1, 2], ["feature 1", "feature 2", "bias"])
    fig.colorbar(im, ax=axes[1, 1])
    fig.tight_layout()
    return fig


def _comparison_plot(experiments: list[Experiment]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for experiment in experiments[-2:]:
        h = experiment.history
        axes[0].plot(h.epochs, h.validation_loss, label=experiment.name)
        axes[1].plot(h.epochs, np.array(h.validation_accuracy) * 100, label=experiment.name)
    axes[0].set(title="Validation loss comparison", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation accuracy comparison", xlabel="epoch", ylabel="percent")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    return fig


class XRayDashboard:
    """Stateful UI; training starts only through the Train button."""
    def __init__(self) -> None:
        self.dataset = widgets.Dropdown(options=["XOR", "Two Moons", "Concentric Circles", "Gaussian Blobs", "Spiral"], value="Two Moons", description="Dataset")
        self.samples = widgets.IntSlider(value=800, min=500, max=2000, step=100, description="Samples", continuous_update=False)
        self.noise = widgets.FloatSlider(value=.12, min=.0, max=.45, step=.01, description="Noise", continuous_update=False)
        self.separation = widgets.FloatSlider(value=1.5, min=.5, max=3.0, step=.1, description="Separation", continuous_update=False)
        self.seed = widgets.IntText(value=42, description="Random seed")
        self.arch = widgets.Text(value="8, 8", description="Hidden layers")
        self.activation = widgets.Dropdown(options=["tanh", "relu", "sigmoid"], value="tanh", description="Activation")
        self.learning_rate = widgets.FloatLogSlider(value=.1, base=10, min=-3, max=0, step=.05, description="Learning rate", readout_format=".4f")
        self.epochs = widgets.IntSlider(value=1500, min=100, max=5000, step=100, description="Epochs", continuous_update=False)
        self.batch = widgets.Dropdown(options=[("Full batch", 0), ("32", 32), ("64", 64), ("128", 128)], value=64, description="Batch size")
        self.l2 = widgets.FloatLogSlider(value=1e-4, base=10, min=-6, max=-1, step=.25, description="L2 penalty", readout_format=".6f")
        self.neuron = widgets.IntSlider(value=1, min=1, max=8, description="Inspect neuron", continuous_update=False)
        self.regenerate = widgets.Button(description="Regenerate dataset", icon="refresh")
        self.train = widgets.Button(description="Train experiment", button_style="success", icon="play")
        self.reset = widgets.Button(description="Reset results", button_style="warning", icon="undo")
        self.check = widgets.Button(description="Run gradient check", icon="check")
        self.output = widgets.Output(layout=widgets.Layout(border="1px solid #d1d5db", padding="8px"))
        self.data: DataSplit | None = None
        self.experiments: list[Experiment] = []
        self.regenerate.on_click(self._regenerate)
        self.train.on_click(self._train)
        self.reset.on_click(self._reset)
        self.check.on_click(self._gradient_check)
        self.neuron.observe(self._show_neuron, names="value")
        self._make_data()

    def _config(self) -> dict:
        return {"dataset": self.dataset.value, "samples": self.samples.value, "noise": self.noise.value, "separation": self.separation.value, "seed": self.seed.value,
                "architecture": _architecture(self.arch.value), "activation": self.activation.value, "learning_rate": self.learning_rate.value,
                "epochs": self.epochs.value, "batch_size": self.batch.value, "l2": self.l2.value}

    def _make_data(self) -> None:
        x, y = generate_dataset(self.dataset.value, self.samples.value, self.noise.value, self.separation.value, self.seed.value)
        self.data = split_and_scale(x, y, self.seed.value)

    def _show(self, *figures) -> None:
        self.output.clear_output(wait=True)
        with self.output:
            for figure in figures:
                if isinstance(figure, str): print(figure)
                else:
                    display(figure); plt.close(figure)

    def _regenerate(self, _=None) -> None:
        try:
            self._make_data()
            self._show(_dataset_plot(self.data, self.dataset.value, self.seed.value), "Dataset regenerated. Changing controls does not retrain until Train experiment is clicked.")
        except Exception as error: self._show(f"Dataset error: {error}")

    def _train(self, _=None) -> None:
        self.train.disabled = True
        try:
            config = self._config()
            self._make_data()  # data/split always reflects visible controls before fit
            self.neuron.max = config["architecture"][0]; self.neuron.value = min(self.neuron.value, self.neuron.max)
            model = BinaryMLP(config["architecture"], config["activation"], config["learning_rate"], config["l2"], config["seed"])
            record_every = max(1, config["epochs"] // 150)
            history = model.fit(self.data.x_train, self.data.y_train, self.data.x_validation, self.data.y_validation, config["epochs"], config["batch_size"], record_every)
            experiment = Experiment(f"Run {len(self.experiments)+1}: {config['activation']} {config['architecture']}", config, self.data, model, history,
                                    evaluate(model, self.data.x_train, self.data.y_train), evaluate(model, self.data.x_validation, self.data.y_validation), evaluate(model, self.data.x_test, self.data.y_test))
            self.experiments.append(experiment)
            message = (f"Training complete in {history.seconds:.2f}s | parameters: {model.parameter_count}\n"
                       f"Train accuracy: {experiment.train_metrics['accuracy']:.1%} | Validation accuracy: {experiment.validation_metrics['accuracy']:.1%}\n"
                       f"Final held-out test accuracy: {experiment.test_metrics['accuracy']:.1%}; precision={experiment.test_metrics['precision']:.3f}, recall={experiment.test_metrics['recall']:.3f}, F1={experiment.test_metrics['f1']:.3f}\n"
                       f"Test confusion matrix [[TN, FP], [FN, TP]]:\n{experiment.test_metrics['confusion_matrix']}\n"
                       "Test data is reported only after final training; do not use it to choose settings. A training/validation gap can indicate overfitting.")
            figures = [_dataset_plot(self.data, self.dataset.value, self.seed.value), _boundary_plot(experiment), _training_plot(history), _internals_plot(experiment, self.neuron.value-1)]
            if len(self.experiments) >= 2: figures.append(_comparison_plot(self.experiments))
            self._show(*figures, message)
        except Exception as error: self._show(f"Training error: {type(error).__name__}: {error}")
        finally: self.train.disabled = False

    def _reset(self, _=None) -> None:
        self.experiments.clear(); self._make_data(); self._show(_dataset_plot(self.data, self.dataset.value, self.seed.value), "Results reset. Dataset is ready; train when you are ready.")

    def _gradient_check(self, _=None) -> None:
        result = gradient_check()
        self._show(f"Gradient check: {'PASS' if result['passed'] else 'FAIL'} | max relative error={result['relative_error']:.3e} | tolerance={result['tolerance']:.1e}")

    def _show_neuron(self, _=None) -> None:
        """The inspector is visualization-only: it never retrains the model."""
        if self.experiments:
            self._show(_internals_plot(self.experiments[-1], self.neuron.value - 1),
                       f"Inspecting neuron {self.neuron.value} from {self.experiments[-1].name}. No retraining occurred.")

    def widget(self):
        controls = widgets.VBox([
            widgets.HTML("<h2> Neural Network X-Ray Lab</h2><p>Interactive NumPy neural-network visualizer.</p>"),
            widgets.HTML("<h3>A. Experiment controls</h3>"),
            widgets.HBox([widgets.VBox([self.dataset, self.samples, self.noise, self.separation, self.seed]), widgets.VBox([self.arch, self.activation, self.learning_rate, self.epochs, self.batch, self.l2, self.neuron])]),
            widgets.HBox([self.regenerate, self.train, self.reset, self.check]),
            widgets.HTML("<p>Regenerate to preview data. Train again after changing settings to compare the latest two runs.</p>"),
            self.output,
        ])
        self._regenerate()
        return controls


def launch_dashboard():
    """Display and return the dashboard, retaining a Python handle for inspection."""
    dashboard = XRayDashboard()
    display(dashboard.widget())
    return dashboard
