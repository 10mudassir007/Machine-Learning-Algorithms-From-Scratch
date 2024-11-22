import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self,n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self,X,y):
        self.X = X
        self.y = y

    # def predict_one(self,x):
    #     distance_func = lambda x1,x2:np.sqrt(np.sum((x1-x2)**2))
    #     distances = [distance_func(x,x_train) for x_train in self.X]
    #     indices = np.argsort(distances)[:self.n_neighbors]
    #     labels = [self.y[i] for i in indices]
    #     most_common = Counter(labels).most_common()
    #     return most_common[0][0]
    
    # def predict(self,X):
    #     preds = [self.predict_one(x) for x in X]
    #     return preds
    
    
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


#sqrt()

#def predict(self,X):
#   for x in X:
#       distances = [distance_func(x,x_train) for x_train in self.X]
#       indices = np.argsort(distances)[:self.n_neighbors]
#       labels = [self.y_train[i] for i in indices]
#       most_common = Counter(labels).most_common()