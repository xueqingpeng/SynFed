import numpy as np
from sklearn.metrics import (accuracy_score, explained_variance_score,
                             f1_score, max_error, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)

from src.logger import get_logger

logger = get_logger(__name__)


TASK_ONE = {"classification": "F1 Score (macro)",
            "regression": "Mean Squared Error"}


class BaseScorer:
    def __init__(self):
        self.metrics = {}

    def cal_scores(self, y_true, y_pred):
        """
        Calculate all metrics scores.
        """
        y_true, y_pred = self.preprocess_pred(y_true, y_pred)

        results = {}
        for metric_name, metric_func in self.metrics.items():
            # Calculate raw score
            score = metric_func(y_true, y_pred)
            results[metric_name] = score
        return results

    def normalize_score(self, score):
        """
        Normalize the metric score to be in the range [0, 1].
        Using the sigmoid function to map any value to the range [0, 1].
        """
        return 1 / (1 + np.exp(-score))  # Sigmoid function to map score to [0, 1]

    def __call__(self, y_true, y_pred):
        """
        Use the selected metric from TASK_ONE and normalize the score.
        If the metric is not found, it will throw an error.
        """
        y_true, y_pred = self.preprocess_pred(y_true, y_pred)

        metric_name = TASK_ONE[self.task_type]
        score = self.metrics[metric_name](y_true, y_pred)
        # Normalize the score
        score = self.normalize_score(score)
        return score

    def preprocess_pred(self, y_true, y_pred):
        if self.task_type == "classification":
            y_pred = list(map(lambda x: -1 if x is None else x, y_pred))
        return y_true, y_pred


class ClassificationScorer(BaseScorer):
    def __init__(self):
        super().__init__()
        self.metrics = {
            'Accuracy': accuracy_score,
            'Precision (micro)': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro'),
            'Recall (micro)': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro'),
            'F1 Score (micro)': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'),
            'Precision (macro)': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
            'Recall (macro)': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
            'F1 Score (macro)': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            # 'ROC AUC': lambda y_true, y_pred_proba: roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        }

        self.task_type = "classification"


class RegressionScorer(BaseScorer):
    def __init__(self):
        super().__init__()
        self.metrics = {
            'Mean Squared Error': mean_squared_error,
            'Mean Absolute Error': mean_absolute_error,
            'R2 Score': r2_score,
            'Mean Absolute Percentage Error': mean_absolute_percentage_error,
            'Median Absolute Error': median_absolute_error,
            'Explained Variance Score': explained_variance_score,
            'Max Error': max_error
        }
        self.task_type = "classification"


def get_scorer(task_type):
    if task_type == "classification":
        return ClassificationScorer()
    elif task_type == "regression":
        return RegressionScorer()
    else:
        raise ValueError(f"Task type '{task_type}' is not supported.")
