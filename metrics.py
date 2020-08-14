import numpy as np
from sklearn import metrics as skmetrics


class ClassificationMetrics():
    def __init__(self):
        self.metrics = {
            'accuracy': self._accuracy,
            'precision': self._precision,
            'recall': self._recall,
            'f1': self._f1,
            'auc': self._auc,
            'logloss': self._logloss,
            'tp': self._true_positive,
            'tn': self._true_negative,
            'fp': self._false_positive,
            'fn': self._false_negative
        }

    def __call__(self, metric, y_true, y_pred, y_proba=None, average="binary"):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        if metric == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be none for auc")
        elif metric == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be none for logloss")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred, average=average)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred, _):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred, average):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred, average=average)

    @staticmethod
    def _recall(y_true, y_pred, average):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred, average=average)

    @staticmethod
    def _precision(y_true, y_pred, average):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred, average=average)

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _true_positive(y_true, y_pred):
        tp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
        return tp

    @staticmethod
    def _true_negative(y_true, y_pred):
        tn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 0:
                tn += 1
        return tn

    @staticmethod
    def _false_positive(y_true, y_pred):
        fp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 1:
                fp += 1
        return fp

    @staticmethod
    def _false_negative(y_true, y_pred):
        fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 0:
                fn += 1
        return fn


'''
    @staticmethod
    def __tpr(y_true, y_pred):
        return self.__recall(y_true, y_pred)

    @staticmethod
    def __fpr(y_true, y_pred):
        return __false_positive(y_true, y_pred) / (__false_positive(y_true.y_pred) + __true_negative(y_true, y_pred))

'''


class RegressionMetrics():
    def __init__(self):
        self.metrics = {
            'mae': 'Mean Absolute Error',
            'mse': 'Mean Squared Error',
            'rmse': 'Root Mean Squared Error',
            'msle': 'Mean Squared Logarithamic Error',
            'rmsle': 'Root Mean Squared Logarithmic Error',
            'mpe': 'Mean Percentage Error',
            'mape': 'Mean Absolute Percentage Error'
        }

    def __call__(self, metric, y_true, y_pred):
        function_mapping = {
            'mae': self._mean_absolute_error(y_true, y_pred),
            'mse': self._mean_squared_error(y_true, y_pred),
            'rmse': self._root_mean_squared_error(y_true, y_pred),
            'msle': self._mean_squared_logarithmic_error(y_true, y_pred),
            'rmsle': self._root_mean_squared_logarithmin_error(y_true, y_pred),
            'mpe': self._mean_percentage_error(y_true, y_pred),
            'mape': self._mean_absolute_persentage_error(y_true, y_pred)
        }
        if metric in function_mapping:
            return function_mapping[metric]
        else:
            raise Exception(
                "Metric not found.\nCheck RegressionMetrics.metrics  for available metrics")

    @staticmethod
    def _mean_absolute_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt-yp)
        return error/len(y_true)

    @staticmethod
    def _mean_squared_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt-yp)**2
        return error/len(y_true)

    @staticmethod
    def _root_mean_squared_error(y_true, y_pred):
        return 0  # np.sqrt(self._mean_squared_error(y_true, y_pred))

    @staticmethod
    def _mean_squared_logarithmic_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += (np.log(1+yt)-np.log(1+yp))**2
        return error/len(y_true)

    @staticmethod
    def _root_mean_squared_logarithmin_error(y_true, y_pred):
        # np.sqrt(self._mean_squared_logarithmic_error(y_true, y_pred))
        return 0

    @staticmethod
    def _mean_percentage_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += (yt-yp)/yt
        return error/len(y_true)

    @staticmethod
    def _mean_absolute_persentage_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt-yp)/yt
        return error/len(y_true)
