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
        for _ in self.n_resample:
            sample = resample(np.arange(batch), replace=True, n_samples=batch)
            sample_data = [d[sample] for d in data]
            res = self.fn(*sample_data)

            self.entries.append(res)

        #aggregate
        self.avg = np.mean(self.entries)
        self.err = stats.sem(self.entries)

        self.ci = [float(n) for n in stats.t.interval(confidence=.95, df=len(self.entries)-1, loc=self.avg, scale=self.err)]

    def __str__(self):
        return f"{self.name}: {self.avg:.2f} ({self.ci[0]:.2f}, {self.ci[1]:.2f})"