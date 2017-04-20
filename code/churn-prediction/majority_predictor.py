from churn_data import ChurnData
import numpy as np

class MajorityPredictor:
    def fit(self, X, y):
        self.mean = np.mean(y).item()
        self.mode = np.round(self.mean)

    def predict(self, X):
        return np.array([self.mode] * len(X))

    def predict_proba(self, X):
        return np.array([[1-self.mean, self.mean]] * len(X))

def main():
    data = ChurnData()
    model = MajorityPredictor()
    model.fit(**data.train)
    data.getScore(model)

if __name__ == "__main__": main()
