"""The same declared objective for screening, selection, confirmation and validation."""
import numpy as np
from sklearn.metrics import f1_score, recall_score

METRICS = {'accuracy', 'balanced_accuracy', 'f1', 'macro_f1', 'weighted_f1', 'micro_f1'}


class Objective:
    def __init__(self, spec, classes):
        self.metric = spec['metric']
        self.classes = np.asarray(classes, dtype=str)
        self.positive = None
        self.threshold = None
        if self.metric not in METRICS:
            raise ValueError(f'Unsupported training objective: {self.metric}')
        if self.metric == 'f1':
            pos = spec.get('positive_class')
            if len(self.classes) != 2 or pos is None or str(pos) not in self.classes:
                raise ValueError('Positive-class F1 needs an explicit positive label and two classes')
            self.positive = self.classes.tolist().index(str(pos))

    def labels(self, probabilities, threshold=None):
        if threshold is None:
            return probabilities.argmax(axis=1)
        if self.positive is None:
            raise ValueError('Only positive-class F1 has a tunable threshold')
        return np.where(probabilities[:, self.positive] >= threshold, self.positive, 1 - self.positive)

    def score(self, actual, predicted):
        if self.metric == 'accuracy':
            return float(np.mean(actual == predicted))
        indices = np.arange(len(self.classes))
        if self.metric == 'balanced_accuracy':
            present = np.unique(actual)
            return float(recall_score(actual, predicted, labels=present, average='macro', zero_division=0))
        if self.metric == 'f1':
            return float(f1_score(actual, predicted, pos_label=self.positive, zero_division=0))
        return float(f1_score(actual, predicted, labels=indices, average=self.metric.split('_')[0], zero_division=0))


def deployed_indices(bundle, p):
    threshold = bundle.get('decision_threshold')
    if threshold is None:
        return p.argmax(axis=1)
    return Objective(bundle['spec'], bundle['classes']).labels(p, threshold)
