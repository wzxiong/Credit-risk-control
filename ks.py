def cal_ks(y_true, y_prob, n_bins=20):
    percentile = np.linspace(0, 100, n_bins + 1).tolist()
    bins = [np.percentile(y_prob, i) for i in percentile]
    bins[0] = bins[0] - 0.01
    bins[-1] = bins[-1] + 0.01
    binids = np.digitize(y_prob, bins) - 1
    y_1 = sum(y_true == 1)
    y_0 = sum(y_true == 0)
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    bin_false = bin_total - bin_true
    true_pdf = bin_true / y_1
    false_pdf = bin_false / y_0
    true_cdf = np.cumsum(true_pdf)
    false_cdf = np.cumsum(false_pdf)
    ks_list = np.abs(true_cdf - false_cdf).tolist()
    ks = max(ks_list)
    return ks, ks_list
