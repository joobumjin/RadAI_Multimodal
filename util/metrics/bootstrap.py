from typing import Optional

import torch
import numpy as np
from sklearn.utils import resample
import scipy.stats as stats

class Bootstrap:
    def __init__(self, name: str, fn: callable, stratify: Optional[callable], n_resample=999, random_state: Optional[int] = None):
        self.name = name

        self.entries = []
        self.avg = None
        self.err = None
        self.ci = None
        self.fn = fn
        self.stratify = stratify
        self.n_resample = n_resample
        self.rand_state = random_state

    def add(self, *data):
        batch = len(data[0])
        strat = self.stratify(data) if self.stratify is not None else None

        #calculate statistic for each resample
        for _ in range(self.n_resample):
            sample = resample(np.arange(batch, dtype=int), replace=True, n_samples=batch, stratify=strat, random_state=self.rand_state).astype(int)
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

def bootstrap_cind_auc(args, test_samples, ci_res, auc_res, n_resample=200):
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.metrics.auc import Auc

    DAYS_IN_YR = 365.2425
    strat = lambda data: data[1] > args.survival_years * DAYS_IN_YR

    def cind(death, times, preds):
        death = death.bool()

        cindex = ConcordanceIndex()
        c = cindex(preds, death, times)
        return c

    def auc(death, times, preds):
        death = death.bool()

        auc = Auc()
        new_time = torch.ones((1,), device=death.device) * args.survival_years * DAYS_IN_YR
        a = auc(preds, death, times, new_time = new_time)
        return a
    
    # ci_res = stats.bootstrap(test_samples, cind, vectorized=False, paired=True, bootstrap_result=ci_res, n_resamples=n_resample, method='percentile', rng=args.seed)
    # auc_res = stats.bootstrap(test_samples, auc, vectorized=False, paired=True, bootstrap_result=auc_res, n_resamples=n_resample, method='percentile', rng=args.seed)
    
    if ci_res is None:
        ci_res = Bootstrap("C-Index", cind, stratify=strat, n_resample=n_resample, random_state=args.seed)
        auc_res = Bootstrap("AUC", auc, stratify=strat, n_resample=n_resample, random_state=args.seed)

    ci_res.add(*test_samples)
    auc_res.add(*test_samples)

    return ci_res, auc_res