import numpy as np
from scipy.stats import wilcoxon

# input final test accuracies for each individual fold after testing
baseline_acc = np.array([
    15.3282, # Fold 1 Test Acc %
    15.2850, # Fold 2 Test Acc % (best baseline fold)
    15.2850, # Fold 3 Test Acc %
    14.5509, # Fold 4 Test Acc %
    14.1623,  # Fold 5 Test Acc %
    15.1552, # Fold 1 Test Acc %
    14.8100, # Fold 2 Test Acc % (best baseline fold)
    14.4646, # Fold 3 Test Acc %
    15.2850, # Fold 4 Test Acc %
    15.9326  # Fold 5 Test Acc %
])

augmented_acc = np.array([
    15.7168, # Fold 1 Test Acc %
    15.4577, # Fold 2 Test Acc % (best augmented fold)
    15.9758, # Fold 3 Test Acc %
    15.5872, # Fold 4 Test Acc %
    14.4214,  # Fold 5 Test Acc %
    17.2280, # Fold 1 Test Acc %
    15.4145, # Fold 2 Test Acc % (best augmented fold)
    16.2781, # Fold 3 Test Acc %
    14.5509, # Fold 4 Test Acc %
    14.2055  # Fold 5 Test Acc %
])


#paired Wilcoxon signed-rank test
statistic, p_value = wilcoxon(
    augmented_acc, 
    baseline_acc, 
    alternative='greater'
)

#result prints
print("WILCOXON SIGNED-RANK TEST RESULTS")
print(f"Baseline Mean Accuracy  : {np.mean(baseline_acc):.4f}%")
print(f"Augmented Mean Accuracy : {np.mean(augmented_acc):.4f}%")
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
