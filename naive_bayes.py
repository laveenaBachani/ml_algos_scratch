import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

X,y = datasets.make_classification(n_samples =1000, n_features = 10, n_classes = 2, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2 , random_state = 42)



class naive_bayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(np.unique(y))
        
        self._mean = np.zeros((num_classes, n_features), dtype = np.float64)
        self._var = np.zeros((num_classes,n_features),dtype = np.float64)
        
        self._priors = np.zeros(num_classes, dtype = np.float64)
        
        for c in self.classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.mean(axis=0)
            self._priors[c] = X_c.shape[0]/n_samples
            
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        
    def _predict(self, x):
        posteriors = []
        
        for index,c in enumerate(self.classes):
            prior = np.log(self._priors[c])
            class_conditional = np.sum(np.log(self._pdf(index,x)))
            posterior = prior+class_conditional
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]
            
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/2*var)
        
        denominator = np.sqrt(2 )
        return numerator/denominator
            
            
model = naive_bayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)