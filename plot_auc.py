import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Setup local paths
model_save_dir = Path("models")
model_save_dir.mkdir(exist_ok=True)

baseline_file = model_save_dir / "resnet50_baseline_eval_data.npz"
augmented_file = model_save_dir / "resnet50_augmented_eval_data.npz"

def compute_macro_roc(eval_data_path, prob_key):
    data = np.load(eval_data_path)
    targets = data['targets']
    
    if prob_key not in data:
        raise KeyError(f"Requested key '{prob_key}' not found in {eval_data_path.name}. Available keys: {data.files}")
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
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs_filtered[:, i])
        
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc

plt.figure(figsize=(9, 7))

#baseline results
if baseline_file.exists():
    print(f"Processing baseline data from: {baseline_file.name}")
    
    # Plot Baseline Ensemble using solid line
    base_fpr, base_tpr, base_auc = compute_macro_roc(baseline_file, 'ensemble_probs')
    plt.plot(base_fpr, base_tpr, label=f'Baseline Ensemble (Macro-AUC = {base_auc:.4f})', color='#e74c3c', lw=2)
    
    # Plot Baseline Best Fold using dotted line
    bf_fpr, bf_tpr, bf_auc = compute_macro_roc(baseline_file, 'best_fold_probs') 
    plt.plot(bf_fpr, bf_tpr, label=f'Baseline Best Fold [F5] (Macro-AUC = {bf_auc:.4f})', color='#e74c3c', linestyle=':', lw=1.5)
else:
    print(f"Warning: Missing baseline data at {baseline_file}")

#augmented results
if augmented_file.exists():
    print(f"Processing augmented data from: {augmented_file.name}")
    
    # Plot Augmented Ensemble using solid line
    aug_fpr, aug_tpr, aug_auc = compute_macro_roc(augmented_file, 'ensemble_probs')
    plt.plot(aug_fpr, aug_tpr, label=f'Augmented Ensemble (Macro-AUC = {aug_auc:.4f})', color='#2ecc71', lw=2)
        
    # Plot Augmented Best Fold with dotted line
    af_fpr, af_tpr, af_auc = compute_macro_roc(augmented_file, 'best_fold_probs')
    plt.plot(af_fpr, af_tpr, label=f'Augmented Best Fold [F2] (Macro-AUC = {af_auc:.4f})', color='#2ecc71', linestyle=':', lw=1.5)
else:
    print(f"Warning: Missing augmented data at {augmented_file}")

#graph plotting
plt.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', label='Random Guess (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
plt.title('PlantDoc Test: Multi-Class ROC-AUC Curves', fontsize=13, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Save output
output_image = model_save_dir / "plantdoc_roc_comparison.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"\nSuccess! Comparison graph saved to: {output_image.absolute()}")

plt.show()