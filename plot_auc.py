import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Setup file paths (Assuming you are running this in Colab, otherwise adjust paths)
model_save_dir = Path("/content/models")
baseline_file = model_save_dir / "resnet50_baseline_eval_data.npz"
augmented_file = model_save_dir / "resnet50_augmented_eval_data.npz"

def compute_macro_roc(eval_data_path):
    # 1. Load saved numpy arrays
    data = np.load(eval_data_path)
    targets = data['targets']
    probs = data['ensemble_probs']  # Plotting the ensemble performance

    # 2. Extract only classes that actually appeared in the test targets
    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)
    
    # Binarize targets for One-vs-Rest calculation
    y_test_binarized = label_binarize(targets, classes=unique_classes)
    
    # Filter, clean and normalize probabilities for present classes
    probs_filtered = probs[:, unique_classes]
    row_sums = probs_filtered.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-9  # Prevent division by zero
    probs_filtered = probs_filtered / row_sums

    # 3. Calculate ROC curve metrics per class
    fpr = {}
    tpr = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs_filtered[:, i])
        
    # 4. Interpolate and compute the Macro-Average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc

# Plot comparisons if both files exist
plt.figure(figsize=(8, 6))

if baseline_file.exists():
    base_fpr, base_tpr, base_auc = compute_macro_roc(baseline_file)
    plt.plot(base_fpr, base_tpr, label=f'Baseline Ensemble (Macro-AUC = {base_auc:.4f})', color='#e74c3c', lw=2)

if augmented_file.exists():
    aug_fpr, aug_tpr, aug_auc = compute_macro_roc(augmented_file)
    plt.plot(aug_fpr, aug_tpr, label=f'Augmented Ensemble (Macro-AUC = {aug_auc:.4f})', color='#2ecc71', lw=2)

# Graph styling and references
plt.plot([0, 1], [0, 1], 'k--', color='#7f8c8d', label='Random Guess (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
plt.title('PlantDoc Test: Multi-Class ROC-AUC Curves', fontsize=13, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Save high-resolution PNG for your report/thesis
output_image = model_save_dir / "plantdoc_roc_comparison.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Comparison graph saved to: {output_image}")
plt.show()