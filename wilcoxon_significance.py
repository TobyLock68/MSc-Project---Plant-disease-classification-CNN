import numpy as np
from scipy.stats import wilcoxon

# input final test accuracies for each individual fold after testing
baseline_fold_accuracies = np.array([
    15.5009, # Fold 1 Test Acc %
    15.3282, # Fold 2 Test Acc % (best baseline fold)
    15.1986, # Fold 3 Test Acc %
    14.5078, # Fold 4 Test Acc %
    14.2055  # Fold 5 Test Acc %
])

augmented_fold_accuracies = np.array([
    15.7168, # Fold 1 Test Acc %
    15.4577, # Fold 2 Test Acc % (best augmented fold)
    15.9758, # Fold 3 Test Acc %
    15.5872, # Fold 4 Test Acc %
    14.4214  # Fold 5 Test Acc %
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
