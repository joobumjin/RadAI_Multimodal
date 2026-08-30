from typing import Optional

import torch
import numpy as np
from sklearn.utils import resample
import scipy.stats as stats

from util.consts import * 
from .c_index import cind, auc

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
        seeds = np.random.SeedSequence(self.rand_state).generate_state(self.n_resample)

        #calculate statistic for each resample
        for seed in seeds:
            sample = resample(np.arange(batch, dtype=int), replace=True, n_samples=batch, stratify=strat, random_state=seed).astype(int)
            sample_data = [d[sample] for d in data]
            res = self.fn(*sample_data)
            self.entries.append(res)

        #aggregate
        self.avg = np.mean(self.entries)
        self.err = stats.sem(self.entries)

        self.ci = [float(n.item()) for n in stats.t.interval(confidence=.95, df=len(self.entries)-1, loc=self.avg, scale=self.err)]

    def __str__(self):
        return f"{self.name}: {self.avg:.3f} ({self.ci[0]:.3f}, {self.ci[1]:.3f})"

def bootstrap_cind_auc(args, test_samples, ci_res, auc_res, n_resample=100):
    def strat(data):
        yrs = data[1] > args.survival_years * DAYS_IN_YR
        return data[0] * 2 + yrs #binary encode
    
    if ci_res is None:
        ci_res = Bootstrap("C-Index", cind, stratify=strat, n_resample=n_resample, random_state=args.seed)
        auc_res = Bootstrap("AUC", lambda *data: auc(*data, args.survival_years), stratify=strat, n_resample=n_resample, random_state=args.seed)

    ci_res.add(*test_samples)
    auc_res.add(*test_samples)

    return ci_res, auc_res