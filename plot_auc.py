from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

# Setup local paths
model_save_dir = Path("models")
model_save_dir.mkdir(exist_ok=True)

# Define file paths for both runs
files = {
    # Run 1
    "Baseline Run 1": {
        "path": model_save_dir / "resnet50_baseline_eval_data.npz",
        "color": "#c0392b",  # Dark Red
        "style_ens": "-",  # Solid
        "style_fold": ":",  # Dotted
    },
    "Augmented Run 1": {
        "path": model_save_dir / "resnet50_augmented_eval_data.npz",
        "color": "#27ae60",  # Dark Green
        "style_ens": "-",  # Solid
        "style_fold": ":",  # Dotted
    },
    # Run 2
    "Baseline Run 2": {
        "path": model_save_dir / "resnet50_baseline_eval_data_pt2.npz",
        "color": "#e67e22",  # Orange
        "style_ens": "--",  # Dashed
        "style_fold": "-.",  # Dash-Dot
    },
    "Augmented Run 2": {
        "path": model_save_dir / "resnet50_augmented_eval_data_pt2.npz",
        "color": "#16a085",  # Teal/Light Green
        "style_ens": "--",  # Dashed
        "style_fold": "-.",  # Dash-Dot
    },
}


def compute_macro_roc(eval_data_path, prob_key):
    data = np.load(eval_data_path)
    targets = data["targets"]

    if prob_key not in data:
        raise KeyError(
            f"Requested key '{prob_key}' not found in {eval_data_path.name}. Available keys: {data.files}"
        )
    probs = data[prob_key]

    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)

    y_test_binarized = label_binarize(targets, classes=unique_classes)

    probs_filtered = probs[:, unique_classes]
    row_sums = probs_filtered.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-9  # Prevent division by zero
    probs_filtered = probs_filtered / row_sums

    fpr = {}
    tpr = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            y_test_binarized[:, i], probs_filtered[:, i]
        )

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc


plt.figure(figsize=(10, 8))

# Loop over all 4 experiment configurations
for label_prefix, config in files.items():
    file_path = config["path"]

    if file_path.exists():
        print(f"Processing {label_prefix} from: {file_path.name}")

        # Plot Ensemble ROC
        ens_fpr, ens_tpr, ens_auc = compute_macro_roc(
            file_path, "ensemble_probs"
        )
        plt.plot(
            ens_fpr,
            ens_tpr,
            label=f"{label_prefix} Ensemble (AUC = {ens_auc:.4f})",
            color=config["color"],
            linestyle=config["style_ens"],
            lw=2,
        )

        # Plot Best Fold ROC
        bf_fpr, bf_tpr, bf_auc = compute_macro_roc(
            file_path, "best_fold_probs"
        )
        plt.plot(
            bf_fpr,
            bf_tpr,
            label=f"{label_prefix} Best Fold (AUC = {bf_auc:.4f})",
            color=config["color"],
            linestyle=config["style_fold"],
            lw=1.5,
        )
    else:
        print(f"Warning: Missing data file for {label_prefix} at {file_path}")

# Random guess line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="#7f8c8d",
    label="Random Guess (AUC = 0.50)",
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.title(
    "PlantDoc Test: Multi-Class ROC-AUC Curves Across Runs",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
plt.grid(True, linestyle=":", alpha=0.6)

# Save output
output_image = model_save_dir / "plantdoc_roc_comparison_4runs.png"
plt.savefig(output_image, dpi=300, bbox_inches="tight")
print(f"\nSuccess! Comparison graph saved to: {output_image.absolute()}")

plt.show()
