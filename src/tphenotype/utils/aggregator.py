class MetricAggregator:
    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics={}):
        for k, v in metrics.items():
            old_v = self.metrics.get(k, 0)
            count = self.counts.get(k, 0)
            new_v = (old_v * count + v) / (count + 1)
            self.counts[k] = count + 1
            self.metrics[k] = new_v

    def query(self, keys=None):
        if keys is not None:
            return {k: self.metrics[k] for k in keys}
        else:
            return self.metrics
