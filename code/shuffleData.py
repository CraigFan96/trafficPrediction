import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from numpy import genfromtxt
Xtr = np.loadtxt(open("../data/nyc_traffic_data_formatted.csv", "rb"), delimiter=",", skiprows=1)
np.random.shuffle(Xtr)
np.save("shuffledData",Xtr)
