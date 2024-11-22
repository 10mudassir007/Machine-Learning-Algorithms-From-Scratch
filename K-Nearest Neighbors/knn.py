import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self,n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X):
        distance_func = lambda x1,x2:np.sqrt(np.sum((x1-x2)**2))
        preds = []

        for x in X:
            distances = [distance_func(x,x_train) for x_train in self.X]
            indices = np.argsort(distances)[:self.n_neighbors]
            labels = [self.y[i] for i in indices]
            most_common = Counter(labels).most_common()[0][0]
            preds.append(most_common)
        return preds
