import torch
import numpy as np
from sklearn.utils import resample
import scipy.stats as stats

class Bootstrap:
    def __init__(self, name: str, fn: callable, n_resample=999):
        self.name = name

        self.entries = []
        self.avg = None
        self.err = None
        self.ci = None
        self.fn = fn
        self.n_resample = n_resample

    def add(self, *data):
        batch = len(data[0])

        #calculate statistic for each resample
        for _ in range(self.n_resample):
            sample = resample(np.arange(batch, dtype=int), replace=True, n_samples=batch, stratify=data[1] < 2 * 365.2425).astype(int)
            sample_data = [d[sample] for d in data]

            # failed = True
            # while failed:
            #     print("trying again")
                # try:
            res = self.fn(*sample_data)
                # except Exception:
                #     print("Error with statistic, resampling")
                #     sample = resample(np.arange(batch, dtype=int), replace=True, n_samples=batch, stratify=data[0]).astype(int)
                #     sample_data = [d[sample] for d in data]
                # else:
                #     failed = torch.isnan(res).any().item()

            self.entries.append(res)

        #aggregate
        self.avg = np.mean(self.entries)
        self.err = stats.sem(self.entries)

        self.ci = [float(n.item()) for n in stats.t.interval(confidence=.95, df=len(self.entries)-1, loc=self.avg, scale=self.err)]

    def __str__(self):
        return f"{self.name}: {self.avg:.3f} ({self.ci[0]:.3f}, {self.ci[1]:.3f})"

def bootstrap_cind_auc(args, test_samples, ci_res, auc_res, n_resample=100):
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.metrics.auc import Auc

    DAYS_IN_YR = 365.2425

    def cind(death, times, preds):
        death = death.bool()

        cindex = ConcordanceIndex()
        c = cindex(preds, death, times)
        return c

    def auc(death, times, preds):
        death = death.bool()

        auc = Auc()
        a = auc(preds, death, times, new_time = torch.ones((1,)) * args.survival_years * DAYS_IN_YR)
        return a
    
    # ci_res = stats.bootstrap(test_samples, cind, vectorized=False, paired=True, bootstrap_result=ci_res, n_resamples=175, method='percentile')
    # auc_res = stats.bootstrap(test_samples, auc, vectorized=False, paired=True, bootstrap_result=auc_res, n_resamples=175, method='percentile')
    
    if ci_res is None:
        ci_res = Bootstrap("C-Index", cind, n_resample=n_resample)
        auc_res = Bootstrap("AUC", auc, n_resample=n_resample)

    ci_res.add(*test_samples)
    auc_res.add(*test_samples)

    return ci_res, auc_res