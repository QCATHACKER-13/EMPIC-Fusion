import numpy as np

# Percentage differences
pp_ke_diffs = [2.04, 4.55, 7.89, 7.08, 5.27, 8.63, 3.76, 0.18, 6.37, 9.39]
prob_fusion_diffs = [5.56, 2.32, 2.85, 2.85, 0.07, 85.96, 54.35, 10.17, 4.43, 22.77]

# Mean and standard deviation
pp_ke_mean_diff = np.mean(pp_ke_diffs)
prob_fusion_mean_diff = np.mean(prob_fusion_diffs)

print("Mean PP's KE Difference:", pp_ke_mean_diff)
print("Mean Prob Fusion Difference:", prob_fusion_mean_diff)

import numpy as np

# Percentage differences
pp_ke_diffs = [2.04, 4.55, 7.89, 7.08, 5.27, 8.63, 3.76, 0.18, 6.37, 9.39]
prob_fusion_diffs = [5.56, 2.32, 2.85, 2.85, 0.07, 85.96, 54.35, 10.17, 4.43, 22.77]

from scipy import stats

# One-sample t-test against a mean of 1%
pp_ke_t_stat, pp_ke_p_val = stats.ttest_1samp(pp_ke_diffs, 1)
prob_fusion_t_stat, prob_fusion_p_val = stats.ttest_1samp(prob_fusion_diffs, 1)

print("PP's KE t-Statistic:", pp_ke_t_stat, "p-Value:", pp_ke_p_val)
print("Prob Fusion t-Statistic:", prob_fusion_t_stat, "p-Value:", prob_fusion_p_val)

