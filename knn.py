
import sklearn.datasets as datasets
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
cmap = ListedColormap(["#FF0000","#00FF00","#0000FF"])
import numpy as np


class knn: 
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, test):
        predicted_labels = [self._predict(x) for x in test]
        return np.array(predicted_labels)
    def _predict(self, x):
        distances = [self.euclidean(x,x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nn = [self.y_train[i] for i in k_indices]
        label = Counter(k_nn).most_common(1)
        return label[0][0]
    def euclidean(self,x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
        
    
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors="k", s =20)
plt.show()
model = knn(k=5)
model.fit(X_train, y_train)
pred =  model.predict(X_test)
print(np.sum(pred == y_test)/len(y_test))
