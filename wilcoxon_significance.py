import numpy as np
from scipy.stats import wilcoxon

# input final test accuracies for each individual fold after testing
baseline_fold_accuracies = np.array([
    34.50, # Fold 1 Test Acc %
    36.20, # Fold 2 Test Acc %
    32.10, # Fold 3 Test Acc %
    35.80, # Fold 4 Test Acc %
    38.90  # Fold 5 Test Acc % (best baseline fold)
])

augmented_fold_accuracies = np.array([
    41.20, # Fold 1 Test Acc %
    45.50, # Fold 2 Test Acc % (best augmented fold)
    40.80, # Fold 3 Test Acc %
    43.10, # Fold 4 Test Acc %
    44.60  # Fold 5 Test Acc %
])

#paired Wilcoxon signed-rank test
statistic, p_value = wilcoxon(
    augmented_fold_accuracies, 
    baseline_fold_accuracies, 
    alternative='greater'
)

#result prints
print("WILCOXON SIGNED-RANK TEST RESULTS")
print(f"Baseline Mean Accuracy  : {np.mean(baseline_fold_accuracies):.4f}%")
print(f"Augmented Mean Accuracy : {np.mean(augmented_fold_accuracies):.4f}%")
print(f"W-Statistic            : {statistic:.1f}")
print(f"p-value                : {p_value:.5f}")

#the greater(>) checks if the augmented model is significantly better than the baseline
alpha = 0.05
if p_value < alpha:
    print(f"\nResult: Statistically SIGNIFICANT improvement (p < {alpha}).")
#Reject the null hypothesis
else:
    print(f"\nResult: NOT statistically significant (p >= {alpha}).")
#Fail to reject the null hypothesis
