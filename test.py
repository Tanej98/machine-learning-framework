import metrics
from sklearn.metrics import accuracy_score
metricname = 'f1'

y_true = [0, 1, 1, 1, 0, 0, 0, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 0]

m = metrics.ClassificationMetrics()

print(m.__calculate__(metricname, y_true, y_pred))
