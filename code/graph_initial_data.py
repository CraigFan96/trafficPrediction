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

def generate_initial_graphs():
	# get lat/long pairs for from and to street positions
	train = np.load("../data/shuffledData.npy")[:,:4]

	for row in train:
		dist =  haversine_distance_formula(row[0], row[1], row[2], row[3])
		plt.plot((row[0], row[2]), (row[1], row[3]), marker = 'o')

	plt.xlim(40,42)
	plt.ylim(-72,-75)
	plt.title('The Streets in Our Dataset')
	plt.xlabel('Latitude')
	plt.ylabel('Longitude')
	plt.savefig("../figures/initial_data.png")

	plt.clf()

	for row in train:
		dist =  haversine_distance_formula(row[0], row[1], row[2], row[3])
		if dist < 5:
			plt.plot((row[0], row[2]), (row[1], row[3]), marker = 'o')

	plt.title('The Streets in Our Dataset (Outliers Removed)')
	plt.xlabel('Latitude')
	plt.ylabel('Longitude')
	plt.savefig("../figures/initial_data_no_outliers.png")