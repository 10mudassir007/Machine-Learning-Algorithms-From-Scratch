import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

df = pd.read_csv("K-Nearest Neighbors/blood.csv")
X = df.drop('Class',axis=1).to_numpy()
y = df['Class'].to_numpy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LogisticRegression()
lr.fit(X_train,y_train)
preds = lr.predict(X_test)

def accuracy(preds,labels):
    return np.mean(preds == labels)

print("Accuracy:",round(accuracy(preds,y_test),3))