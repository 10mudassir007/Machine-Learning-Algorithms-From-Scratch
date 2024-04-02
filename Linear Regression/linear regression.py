import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('House-Price-Predictor-Using-Linear-Regression/housing.csv').head(1000)
df = pd.concat((df.bedrooms,df.price),axis=1)
X = np.array([df.bedrooms])
y = np.array([df.price])

class LinearRegression:
    def __init__(self,lr=0.001,epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self,x,y):
        samples,features = x.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.epochs):
            y_hat = np.dot(x,self.weights) + self.bias

            dw = (1/samples) * np.dot(x.T,(y_hat-y))
            db = (1/samples) * np.sum(y_hat-y)

            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
    def predict(self,x):
        y_pred = np.dot(np.sum(self.weights),x) + self.bias
        return y_pred
    

model = LinearRegression()
model.fit(X,y)
print(model.predict(np.array([1])))
