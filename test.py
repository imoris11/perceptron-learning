import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from perceptron import Perceptron


    
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def _unit_step_func(x):
    return np.where(x>=0, 1, -1)

#find weights and position of elements for 2D data
def find_weights(X):
    n_samples, n_features = X.shape
    weights = [0.01, 0.2]
    bias = 1
    y=[]
    for idx, x_i in enumerate(X):
        linear_output = np.dot(x_i, weights) + bias
        y_predicted = _unit_step_func(linear_output)
        y.append(y_predicted)
    return y, weights

#generate randomly generated linear separable datasets 
X , y = datasets.make_blobs(n_samples=1000, n_features=10, centers=2, cluster_std=1.05, random_state=12)

# y, weights = find_weights(X)
    

    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

p = Perceptron(learning_rate=0.01, n_iters=10)
p.fit(X_train, y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X[:,0], X[:,1],marker='p',c=y)

x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])

x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

# x1_1 = (-weights[0] * x0_1 - bias) / weights[1]
# x1_2 = (-weights[0] * x0_2 - bias) / weights[1]

ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k')

ymin = np.amin(X_train[:,1])
ymax = np.amax(X_train[:,1])
ax.set_ylim([ymin-3,ymax+3])

plt.show()
p.weights
p.bias