import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from matplotlib.colors import ListedColormap
import timeit
from sklearn import datasets, svm
import pandas as pd
from sklearn.model_selection import train_test_split
import psutil
import os

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
start = timeit.default_timer()

y1 = df.iloc[0:100, 4].values
y1=np.where( y1 == 'Iris-setosa',-1 ,1)
x = df.iloc[0:100, [0, 2]].values
pn = Perceptron(0.1, 10)
pn.fit(x, y1)
predictions = pn.predict(x)

print(pn.score(x, y1))


def plot_decision_regions(x, y1, classifier, resolution=0.02):
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y1))])
   x1_min, x1_max = x[:,  0].min() - 1, x[:, 0].max() + 1
   x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   for idx, cl in enumerate(np.unique(y1)):
      plt.scatter(x=x[y1 == cl, 0], y=x[y1 == cl, 1],
      alpha=0.8, c=cmap(idx),
      marker=markers[idx], label=cl)

plot_decision_regions(x, y1, classifier=pn)
plt.xlabel('sepal length')
plt.ylabel('petal length')

stop = timeit.default_timer()


print("CPU Utilization: ")
print(str(psutil.cpu_percent()))

process = psutil.Process(os.getpid())
mem = process.memory_percent()
print("Memory Utilization for this process: ")
print(mem)


print("Running time: ")
print(stop - start)


plt.show()