import pandas as pd
from scipy.stats import ttest_ind_from_stats


def text_to_mean(val):
    val = val.split('+-')
    mean = float(val[0])
    return mean


def text_to_std(val):
    val = val.split('+-')
    std = float(val[1]) / 1.96
    return std


def get_p_value(val, val2, n):
    mean1 = text_to_mean(val)
    std1 = text_to_std(val)
    mean2 = text_to_mean(val2)
    std2 = text_to_std(val2)
    _, p_value = ttest_ind_from_stats(mean1, std1, n, mean2, std2, n)
    return p_value


def find_best_method(results, metrics):
    # test if the best row is statistically best one
    p_values = pd.DataFrame()
    p_values['method'] = results['method']
    for m in metrics:
        means = results[m].apply(text_to_mean)
        best_row = means.argmax()
        best_val = results.loc[best_row, m]
        best_n = int(results.loc[best_row, 'n'])

        for i in results.index:
            val = results.loc[i, m]
            mean1 = text_to_mean(val)
            std1 = text_to_std(val)
            n1 = int(results.loc[i, 'n'])
            mean2 = text_to_mean(best_val)
            std2 = text_to_std(best_val)
            _, p_value = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, best_n)
            p_values.loc[i, m] = p_value
        p_values.loc[best_row, m] = 'best'
    return p_values
