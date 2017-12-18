"""
@authors Craig Fan and Sam Ginzburg

This program reproduces all of our graphs and output.

Preparation Instructions:
1) First run convert data file to convert our original nyc_traffic_data.csv file to the formatted version 
(instructions can be found in convert_data_file.py, google cloud API key required)

2) Run python shuffleData.py, in order to shuffle all the datacases in the file

3) Run python scale_data.py, in order to scale all the hourly throughputs by the lengths of the roads.


Run Instructions (after completing preparation):
python run_me.py

"""


#from multioutput import multioutput
import code
import sys

def keyboard(banner=None):
   '''
    Function that mimics the matlab keyboard command
    Acquired from Ben Marlin.
    '''
   # use exception trick to pick up the current frame
   try:
       raise None
   except:
       frame = sys.exc_info()[2].tb_frame.f_back
   print('# Use quit() to exit :) Happy debugging!')
   # evaluate commands in current namespace
   namespace = frame.f_globals.copy()
   namespace.update(frame.f_locals)
   try:
       code.interact(banner=banner, local=namespace)
   except SystemExit:
       return


import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import Ridge
from math import sqrt
from sklearn.metrics import make_scorer, mean_absolute_error

# import this to run the code to regenerate graphs
from graph_initial_data import generate_initial_graphs
generate_initial_graphs()
np.random.seed(0)


"""
This function generates the errors and computes the minimum parameter that is paired with the error
"""
def setUpPlot(hyperParameterList, regr):
    errors = np.zeros((len(hyperParameterList), 2))
    minimumParameter = 1
    minY = -1
    for x, y in zip(hyperParameterList, range(len(hyperParameterList))):
        errors[y, 0] = sqrt(1 * regr.cv_results_['mean_train_score'][y])
        errors[y, 1] = sqrt(1 * regr.cv_results_['mean_test_score'][y])
        if(errors[y, 1] > minY):
            minY = errors[y, 1]
            minimumParameter = x
    return errors, minimumParameter

"""
This function creates a plot of the errors and  hyperparameters, and then saves the figure to the ../figures/ directory
"""
def createPlot(errors, hyperParameterList, minimumParameter, modelname):
    inds = np.arange(len(hyperParameterList))
    labels = ["Train", "Validation"]
    smallestValue = min(errors[:, 1])
    smallestValueIndex = np.argmin(errors[:, 1])
    plt.figure(10, figsize=(10,8))  #6x4 is the aspect ratio for the plot
    plt.plot(hyperParameterList,errors[:, 0],'or-', linewidth=3) #Plot the first series in red with circle marker
    plt.plot(hyperParameterList,errors[:, 1],'sb-', linewidth=3) #Plot the first series in blue with square marker

    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Mean Absolute Error, " + "Smallest Mean Absolute Error: " + str(smallestValue)) #Y-axis label
    plt.xlabel("Value: " + "Best value: " + str(hyperParameterList[smallestValueIndex])) #X-axis label
    plt.title("Mean Absolute Error vs Value (" + modelname + ")") #Plot title
    plt.legend(labels,loc="best")

    #Make sure labels and titles are inside plot area
    plt.tight_layout()

    plt.savefig("../figures/"+modelname+".png")
    plt.close()

#Xtr = np.loadtxt(open("../data/nyc_traffic_data_formatted.csv", "rb"), delimiter=",", skiprows=1)
#train = np.load("../data/shuffledData.npy")

# This is the scaled dataset that we used for the project
train = np.load("../data/shuffledData_normalized.npy")
Xtr = train[:4000]
Xte = train[4000:]
#Xtr = np.random.shuffle(firstXtr)
#Ytr = theLabels


qualityList = []
for i in range(1, 51):
    kmeans = KMeans(n_clusters = i, random_state = 5)
    kmeans.fit(Xtr)
    predictions = kmeans.predict(Xte)
    quality = kmeans.score(Xte, predictions)
    print quality
    qualityList.append(np.abs(quality))


clusterAmount = range(1, 51)
#inds = np.arange(7)
plt.figure(10, figsize=(10,8))  #6x4 is the aspect ratio for the plot
plt.plot(clusterAmount, qualityList, 'or-', linewidth=3) #Plot the first series in red with circle marker

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Quality value") #Y-axis label
plt.xlabel("Cluster amount") #X-axis label
plt.title("Cluster Amount vs Quality") #Plot title

#Make sure labels and titles are inside plot area
plt.tight_layout()

#plt.show()
plt.close()

#With Elbow method, we get an I amount to be the optimal number
#of clusters

#Assuming 1 is the ideal amount of clusters

optimalClusterNumber = 5

optimizedKMeans = KMeans(n_clusters = optimalClusterNumber, random_state = 5)
optimizedKMeans.fit(Xtr)
optimizedPredictions = optimizedKMeans.predict(Xtr)
optimizedPredictionsXte = optimizedKMeans.predict(Xte)
"""
Now you have the clusters that each data point belongs to
Loop over this data, and create datasets. A dataset per each cluster
With the datasets, train models on those datasets and assign the learned model to
the specified cluster
"""

datasetPerCluster = [[] for i in range(optimalClusterNumber)]
#K being the optimal number of clusters
for i in range(0, optimalClusterNumber):
    for j in range(0, len(optimizedPredictions)):
        if(optimizedPredictions[j] == i):
            datasetPerCluster[i].append(Xtr[j])


datasetPerClusterTest = [[] for i in range(optimalClusterNumber)]
#K being the optimal number of clusters
for i in range(0, optimalClusterNumber):
    for j in range(0, len(optimizedPredictionsXte)):
        if(optimizedPredictionsXte[j] == i):
            datasetPerClusterTest[i].append(Xte[j])
"""
[:19] is the data, [19:] is the labels
"""

for x in range(0, optimalClusterNumber):
    datasetPerCluster[x] = np.array(datasetPerCluster[x])
    datasetPerClusterTest[x] = np.array(datasetPerClusterTest[x])

optimizedModelPerCluster = [[[] for i in range(optimalClusterNumber)] for i in range(3)]
hyperparamlist = [[[] for i in range(optimalClusterNumber)] for i in range(3)]

"""
This is the main loop that we used to train our models on the various clusters we generated previously
We train each model once for each different number of clusters, and generate a graph of the hyperparameter
selection method for that model.
"""
for model_count in range(3):
    for i in range(0, optimalClusterNumber):
        if model_count == 0: # random forest regression
            tuned_parameters = [{'n_estimators': [1, 5, 10, 50]}]
            model = GridSearchCV(RandomForestRegressor(random_state=15), tuned_parameters, cv = 3, n_jobs = -1, scoring=make_scorer(mean_absolute_error))
            model.fit(datasetPerCluster[i][:, :19], datasetPerCluster[i][:, 19:])
            hyperparamlist[model_count][i] = [1, 5, 10, 50]
        elif model_count == 1: # knn regression
            KNeighborsRegressorHyperParameterList = [1, 3, 5, 10, 50]
            tuned_parametersKNeighborsRegressor = [{'n_neighbors': KNeighborsRegressorHyperParameterList}]
            gridSearchForKNRegression = GridSearchCV(KNeighborsRegressor(), tuned_parametersKNeighborsRegressor, cv = 3, n_jobs = -1, scoring=make_scorer(mean_absolute_error))
            gridSearchForKNRegression.fit(datasetPerCluster[0][:, :19], datasetPerCluster[0][:, 19:])
            model = gridSearchForKNRegression
            hyperparamlist[model_count][i] = KNeighborsRegressorHyperParameterList
        elif model_count == 2:
            ridgeParams = [0.01, 0.1, 1, 5, 10, 20, 50]
            model = GridSearchCV(Ridge(normalize=False), {'alpha': ridgeParams}, cv = 3, n_jobs = -1, scoring=make_scorer(mean_absolute_error))
            model = model.fit(datasetPerCluster[i][:, :19], datasetPerCluster[i][:, 19:])
            hyperparamlist[model_count][i] = ridgeParams
        else:
            print "Error, only 3 models currently supported"
            exit()

        optimizedModelPerCluster[model_count][i] = model
        print model.best_params_

for model_count in range(3):
    print model_count
    for i in range(0, optimalClusterNumber):
        print i
        if model_count == 0:
            randomForestSetUP = setUpPlot(hyperparamlist[model_count][i], optimizedModelPerCluster[model_count][i])
            createPlot(randomForestSetUP[0],hyperparamlist[model_count][i], randomForestSetUP[1], "Random Forest Regressor (Cluster Count: " + str(i+1) + ")")
        elif model_count == 1:
            knnSetUp = setUpPlot(hyperparamlist[model_count][i], optimizedModelPerCluster[model_count][i])
            createPlot(knnSetUp[0],hyperparamlist[model_count][i], knnSetUp[1], "KNeighborsRegressor (Cluster Count: " + str(i+1) + ")")
        elif model_count == 2:
            ridgeSetUp = setUpPlot(hyperparamlist[model_count][i], optimizedModelPerCluster[model_count][i])
            createPlot(ridgeSetUp[0],hyperparamlist[model_count][i], ridgeSetUp[1], "Ridge Regression (Cluster Count: " + str(i+1) + ")")
        else:
            print "Error, only 3 models currently supported"
            exit()
