class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1, avg=True):
        if not avg: val /= float(n)
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * float(n)
        self.count += float(n)
        self.avg = float(self.sum) / float(self.count)
