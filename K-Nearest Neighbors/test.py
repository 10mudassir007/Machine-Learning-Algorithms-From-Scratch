import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from knn import KNearestNeighbors
from tqdm import tqdm

df = pd.read_csv("K-Nearest Neighbors/blood.csv")
X = df.drop("Class",axis=1).to_numpy()
y = df['Class'].to_numpy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

accs = []
ks = []
for k in tqdm(range(2,30)):
    clf = KNearestNeighbors(n_neighbors=k)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    corr_preds = 0
    for i,j in zip(preds,y_test):
        if i == j:
            corr_preds += 1
    
    acc = corr_preds / len(y_test)
    accs.append(acc)
    ks.append(k)
plt.scatter(x=ks,y=accs)
plt.plot(ks, accs, label='Line')
plt.show()
max_idx = accs.index(max(accs))
print(f"Max Accuracy: {accs[max_idx]} at k={ks[max_idx]}")