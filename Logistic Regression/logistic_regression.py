import numpy as np

np.seterr(over='ignore', invalid='ignore')

class LogisticRegression:
    def __init__(self,alpha=0.001,iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        #sigmoid(z) = 1 / 1 + exp(-z)
        self.sigmoid_func = lambda x:1 / (1 + np.exp(-x))
        
        training_examples,n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            #z = w * x + b
            z = np.dot(X,self.weights) + self.bias
            pred = self.sigmoid_func(z)

            #dw/dj = 1/n * X.T * difference between labels and predicted or residual error
            dw = (1 / training_examples) * np.dot(X.T,(pred - y))
            #db/dj = 1/n * sum(difference between labels and predicted or residual error)
            db = (1 / training_examples) * np.sum(pred-y)

            #w = w - learning_rate * dw/dj
            self.weights -= (self.alpha * dw)
            #b = b - learning_rate * db/dj
            self.bias -= (self.alpha * db)

    def predict(self,X):
        z = np.dot(X,self.weights) + self.bias
        preds = self.sigmoid_func(z)
        final_preds = np.array([1 if i>0.5 else 0 for i in preds])
        return final_preds
    