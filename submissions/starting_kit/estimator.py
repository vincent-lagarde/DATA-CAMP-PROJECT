from sklearn.dummy import DummyClassifier

class DummyWrapper(DummyClassifier):
    def predict(self, X):
        preds = super().predict(X)
        # Convert predictions to plain Python ints
        return [int(p) for p in preds]

def get_estimator():
    return DummyWrapper(strategy='most_frequent')
