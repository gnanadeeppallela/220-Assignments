from django.shortcuts import render,render_to_response

from pylab import *
import timeit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import pandas as pd
from sklearn.model_selection import train_test_split
import psutil
import os


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from django.http import HttpResponse
# Create your views here.

def loic(request):

    return render(request, 'firstapp/LOIC-download-Detection.html', context=None)


def postsubmit(request):
    if request.method == 'POST':
        df = pd.read_csv(request.FILES['loicfile'], sep=",", error_bad_lines=False, index_col=False, dtype='float64')
        start = timeit.default_timer()
        X_data = np.array(df.ix[0:150, 1:3])
        Y_target = np.array(df.ix[0:150, 5:])

        validation_size = 0.20
        seed = 7

        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_target, test_size=validation_size,
                                                            random_state=seed)
        # clf=svm.SVR(kernel='rbf', gamma=1,C=1)
        clf = svm.SVC(kernel='linear', verbose=True, gamma='auto', C=1.0)

        m = clf.fit(X_train, Y_train.ravel())
        predictions = clf.predict(X_test)

        # print(clf.score(X_test, Y_test))

        np.mean((clf.predict(X_test) - Y_test) ** 2)

        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        h = (x_max / x_min) / 100

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        plt.subplot(1,1,1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, color='red', alpha=0.8)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired)

        plt.ylabel('sepal-width')


        stop = timeit.default_timer()

        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30

        print("CPU Utilization: ")
        print(psutil.cpu_percent())

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
        plt.xlabel('sepal-length')
        plt.title(s)
        #plt.figure(figsize=(20, 10))
        canvas = FigureCanvas(plt.figure(1))
        response = HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

        #return HttpResponse(buffer.getvalue(), mimetype="image/png")

        #return render(request, 'firstapp/result.html', buffer.getvalue(), mimetype="image/png")

    #return render(request, 'Home/result.html', context={'sourceaddress': sourceaddress})
