from typing import Optional
from argparse import Namespace

from torch import Tensor
import numpy as np
from sklearn.utils import resample
import scipy.stats as stats

from util.consts import * 
from .c_index import cind, auc

class Bootstrap:
    """
    Manual Bootstrapping calculation
    Statefully keeps track of average, error, and 95% confidence interval
    """

    def __init__(self, name: str, fn: callable, stratify: Optional[Tensor], n_test: int = 16, n_resample: int = 999, random_state: Optional[int] = None):
        """
        params:
        - name: name of the statistic 
        - fn: the statstic which will be bootstrap analyzed
        - stratify: optional function to stratify data
        - n_resample: how many times to run bootstrap resampling
        """
        self.name = name
        self.entries = []
        self.avg = None
        self.err = None
        self.ci = None
        self.fn = fn
        self.n_resample = n_resample
        self.rand_state = random_state

        seeds = np.random.SeedSequence(random_state).generate_state(self.n_resample)
        self.sample_inds = [resample(np.arange(n_test, dtype=int), replace=True, n_samples=n_test, stratify=stratify, random_state=seed).astype(int) for seed in seeds] #[n_resample, n_test] of just inds

    def add(self, *data):
        """
        Takes data, calculates bootstrap values, adds those bootstrap values to entries, and updates average, error, and 95% CI
        """
        #calculate statistic for each resample
        res = [None] * len(self.sample_inds)
        for i, sample in enumerate(self.sample_inds):
            sample_data = [d[sample] for d in data] 
            res[i] = self.fn(*sample_data) #cant be vectorized :) 
        self.entries += res

        #aggregate
        entries = np.array(self.entries)
        self.avg = np.mean(entries)
        self.err = stats.sem(entries)

        self.ci = [float(n.item()) for n in stats.t.interval(confidence=.95, df=len(self.entries)-1, loc=self.avg, scale=self.err)]

    def __str__(self):
        return f"{self.name}: {self.avg:.3f} ({self.ci[0]:.3f}, {self.ci[1]:.3f})"

def bootstrap_cind_auc(args: Namespace, test_samples: list[Tensor], 
                       ci_res: Optional[Bootstrap] = None, auc_res: Optional[Bootstrap] = None) -> tuple[Bootstrap, Bootstrap]:
    """
    Keeps track of Bootstrap object instances for C-Index and AUC evaluation

    params:
    - args: argparse.Namespace
    - test_samples: compiled test data with structure [deaths, times, preds]
    - ci_res: Bootstrap object instance for C-Index
    - auc_res: Bootstrap object instance for AUC
    """
    def strat(data):
        yrs = data[1] > args.survival_years * DAYS_IN_YR
        return data[0] * 2 + yrs #binary encode
    
    if ci_res is None:
        stratify = strat(test_samples)
        kwargs = {
            "stratify": stratify, 
            "n_resample": args.resample, 
            "n_test": len(test_samples[0]), 
            "random_state": args.seed
        }
        ci_res = Bootstrap("C-Index", cind, **kwargs)
        auc_res = Bootstrap("AUC", lambda *data: auc(*data, args.survival_years), **kwargs)

    ci_res.add(*test_samples)
    auc_res.add(*test_samples)

    return ci_res, auc_res