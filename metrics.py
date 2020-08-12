import numpy as np


class ClassificationMetrics():
    def __init__(self):
        self.metrics = [
            'accuracy', 'precision', 'recall'
        ]

    def __calculate__(self, metric, y_true, y_pred):
        if metric == 'accuracy':
            return self.__accuracy(y_true, y_pred)
        elif metric == 'precision':
            return self.__precision(self.__true_positive(y_true, y_pred), self.__false_positive(y_true, y_pred))
        elif metric == 'recall':
            return self.__recall(self.__true_positive(y_true, y_pred), self.__false_negative(y_true, y_pred))
        elif metric == 'f1':
            p = self.__precision(self.__true_positive(
                y_true, y_pred), self.__false_positive(y_true, y_pred))
            r = self.__recall(self.__true_positive(
                y_true, y_pred), self.__false_negative(y_true, y_pred))
            f1 = 2*p*r/(p+r)
            return f1

    @staticmethod
    def __accuracy(y_true, y_pred):
        correct_count = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == yp:
                correct_count += 1
        return correct_count/len(y_true)

    @staticmethod
    def __true_positive(y_true, y_pred):
        tp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
        return tp

    @staticmethod
    def __true_negative(y_true, y_pred):
        tn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 0:
                tn += 1
        return tn

    @staticmethod
    def __false_positive(y_true, y_pred):
        fp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 1:
                fp += 1
        return fp

    @staticmethod
    def __false_negative(y_true, y_pred):
        fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 0:
                fn += 1
        return fn

    @staticmethod
    def __precision(tp, fp):
        p = tp / (tp+fp)
        return p

    @staticmethod
    def __recall(tp, fn):
        r = tp/(tp+fn)
        return r


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

    def __calculate__(self, metric, y_true, y_pred):
        function_mapping = {
            'mae': self.__mean_absolute_error(y_true, y_pred),
            'mse': self.__mean_squared_error(y_true, y_pred),
            'rmse': self.__root_mean_squared_error(y_true, y_pred),
            'msle': self.__mean_squared_logarithmic_error(y_true, y_pred),
            'rmsle': self.__root_mean_squared_logarithmin_error(y_true, y_pred),
            'mpe': self.__mean_percentage_error(y_true, y_pred),
            'mape': self.__mean_absolute_persentage_error(y_true, y_pred)
        }
        if metric in function_mapping:
            return function_mapping[metric]
        else:
            raise Exception(
                "Metric not found.\nCheck RegressionMetrics.metrics  for available metrics")

    @staticmethod
    def __mean_absolute_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt-yp)
        return error/len(y_true)

    @staticmethod
    def __mean_squared_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt-yp)**2
        return error/len(y_true)

    @staticmethod
    def __root_mean_squared_error(y_true, y_pred):
        return 0  # np.sqrt(self.__mean_squared_error(y_true, y_pred))

    @staticmethod
    def __mean_squared_logarithmic_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += (np.log(1+yt)-np.log(1+yp))**2
        return error/len(y_true)

    @staticmethod
    def __root_mean_squared_logarithmin_error(y_true, y_pred):
        # np.sqrt(self.__mean_squared_logarithmic_error(y_true, y_pred))
        return 0

    @staticmethod
    def __mean_percentage_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += (yt-yp)/yt
        return error/len(y_true)

    @staticmethod
    def __mean_absolute_persentage_error(y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            error += np.abs(yt-yp)/yt
        return error/len(y_true)
