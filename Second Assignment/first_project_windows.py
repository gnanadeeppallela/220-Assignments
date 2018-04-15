import timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import pandas as pd
from sklearn.model_selection import train_test_split
import psutil
import os
import sys

location = r'C:\gnanadeep\iris.csv'
start = timeit.default_timer()
df = pd.read_csv(location, sep=",", error_bad_lines=False, index_col=False, dtype='float64')
X_data = np.array(df.ix[0:99,1:3])
Y_target = np.array(df.ix[0:99,5:])

validation_size = 0.20
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_target, test_size=validation_size, random_state=seed)
#clf=svm.SVR(kernel='rbf', gamma=1,C=1)
clf=svm.SVC(kernel='linear', verbose=True, gamma='auto', C=1.0)

m=clf.fit(X_train,Y_train.ravel())
predictions = clf.predict(X_test)

print(clf.score(X_test, Y_test))

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, color='red', alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired)
plt.xlabel('sepal-length')
plt.ylabel('petal-length')
plt.title('Train data')

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
