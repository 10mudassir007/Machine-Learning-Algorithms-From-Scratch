import numpy as np

class NaiveBayes:

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes,n_features),dtype=np.float64)
        self.var = np.zeros((n_classes,n_features),dtype=np.float64)
        self.priors = np.zeros(n_classes,dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx,:] = X_c.mean(axis=0)
            self.var[idx,:] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0]/float(n_samples)
    
    def predict(self,X):
        y_pred = [self.pred(x) for x in X]
        return np.array(y_pred)
    
    def pred(self,x):
        posteriors = []

        for idx,c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.pdf(idx,x)))
            posterior += prior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    def pdf(self,class_idx,x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator