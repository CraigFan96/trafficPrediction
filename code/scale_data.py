import matplotlib.pyplot as plt
import numpy as np
import math

# equation obtained from https://en.wikipedia.org/wiki/Haversine_formula
def haversine_distance_formula(x1, y1, x2, y2):
	R = 6371
	f1 = x1 * (math.pi/180)
	f2 = x2 * (math.pi/180)
	df = (x2 - x1) * (math.pi/180)
	dl = (y2 - y1) * (math.pi/180)

	a = math.sin(df/2) * math.sin(df/2) + math.cos(f1) * math.cos(f2) * math.sin(dl/2) * math.sin(dl/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	return R * c

# get lat/long pairs for from and to street positions
train = np.load("../data/shuffledData.npy")

# scale the throughput for each street by the distance in km of the road
for row in train:
	dist =  haversine_distance_formula(row[0], row[1], row[2], row[3])
	if dist != 0: # some streets floating points get rounded down to 0 so we don't scale those
		row[-24:] /= dist
	
np.save("../data/shuffledData_normalized", train)