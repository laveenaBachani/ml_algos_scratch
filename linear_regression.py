from turtle import color
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

X, Y  = datasets.make_regression(n_samples = 100, n_features =1, noise = 20, random_state =1234)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

plt.figure()
plt.scatter(X,Y,s=30)
plt.show()





class LinearRegression:
    def __init__(self, lr = 0.1, num_iter = 1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weight = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.num_iter):
            y_pred = np.dot(X,self.weight)+self.bias
            dw = (1/n_samples) * np.dot(X.T, y_pred-y)
            db = (1/n_samples) * np.sum(y_pred-y)
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
        
    def predict(self, X):
        y_predicted = np.dot(X, self.weight)+self.bias
        return y_predicted
   

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)
        
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(mse(y_test,y_pred))

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train,y_train, color= cmap(0.9), s=10)
m2 = plt.scatter(X_test,y_test,color = cmap(0.5), s=10)
plt.plot(X_test,y_pred , color = "black")
plt.show()
