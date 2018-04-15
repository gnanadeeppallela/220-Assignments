from django.shortcuts import render,render_to_response
import numpy as np
from sklearn.metrics import accuracy_score
from pylab import *
import timeit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import pandas as pd
from sklearn.model_selection import train_test_split
import psutil
import os

from matplotlib.colors import ListedColormap

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from django.http import HttpResponse



class Perceptron(object):
   def __init__(self, rate = 0.01, niter = 10):
      self.rate = rate
      self.niter = niter

   def fit(self, X, y):
      self.weight = np.zeros(1 + X.shape[1])

      self.errors = []

      for i in range(self.niter):
         err = 0
         for xi, target in zip(X, y):
            delta_w = self.rate * (target - self.predict(xi))
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w
            err += int(delta_w != 0.0)
         self.errors.append(err)
      return self

   def net_input(self, X):
      return np.dot(X, self.weight[1:]) + self.weight[0]

   def predict(self, X):
      return np.where(self.net_input(X) >= 0.0, 1, -1)

   def score(self, X, y, sample_weight=None):
      return accuracy_score(y, self.predict(X), normalize=True)

def loic(request):

    return render(request, 'firstapp/LOIC-download-Detection.html', context=None)


def postsubmit(request):
    if request.method == 'POST':
        df2=request.POST['loicfile']
        df = pd.read_csv(str(df2), header=None)
        start = timeit.default_timer()

        y1 = df.iloc[0:100, 4].values
        y1 = np.where(y1 == 'Iris-setosa', -1, 1)
        x = df.iloc[0:100, [0, 2]].values
        pn = Perceptron(0.1, 10)
        pn.fit(x, y1)
        predictions = pn.predict(x)

        print(pn.score(x, y1))

        def plot_decision_regions(x, y1, classifier, resolution=0.02):
            markers = ('s', 'x', 'o', '^', 'v')
            colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
            cmap = ListedColormap(colors[:len(np.unique(y1))])
            x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
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

        s0="sepal-length"
        n='\n'
        sd=psutil.cpu_percent()
        s1="CPU Utilization: "
        s2=str(sd)
        s3="Memory Utilization: "
        s4=str(mem)
        s5="Running time: "
        s6=str(stop - start)

        s=s1+s2+n+s3+s4+","+s5+s6
        plt.title(s)
        #plt.figure(figsize=(20, 10))
        canvas = FigureCanvas(plt.figure(1))
        response = HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

        #return HttpResponse(buffer.getvalue(), mimetype="image/png")

        #return render(request, 'firstapp/result.html', buffer.getvalue(), mimetype="image/png")

    #return render(request, 'Home/result.html', context={'sourceaddress': sourceaddress})
