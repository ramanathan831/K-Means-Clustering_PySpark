import sys
import numpy as np
from math import sqrt
import findspark
findspark.init()
from pyspark import SparkContext
import matplotlib.pyplot as plt

def euclidean_distance(point1,point2):
	assert(len(point1) == len(point2))
	distance = 0.0
	index = 0
	while(index < len(point1)):
		distance += (point1[index] - point2[index])**2
		index += 1
	return sqrt(distance)

def assign_centroids(centroids):
	def _assign_centroids(curr_point):
		index = 0
		euc_list = []
		while(index < len(centroids)):
			euc_list.append(euclidean_distance(curr_point, centroids[index]))
			index += 1
		min_dist = min(euc_list)
		cluster_id = euc_list.index(min_dist)
		return curr_point,[cluster_id,(min_dist**2)]
	return _assign_centroids

def compute_new_centroid(points):
	outer_index = 0
	new_centroid = []
	points = list(points)
	while(outer_index < len(points[0])):
		inner_index = 0
		sumvar = 0
		while(inner_index < len(points)):
			sumvar += points[inner_index][outer_index]
			inner_index += 1
		new_centroid.append(sumvar/len(points))
		outer_index += 1
	return new_centroid

def compute_new_centroids(curr_pair):
	new_centroid = compute_new_centroid(curr_pair[1])
	return curr_pair[0], new_centroid

def reverse_key_value(curr_pair):
	return curr_pair[1][0], curr_pair[0]

def main():
	datasetfile = sys.argv[1]
	centroid_method = sys.argv[2]
	if(centroid_method == "1"):
		centroid_file = "data/c1.txt"
	elif(centroid_method == "2"):
		centroid_file = "data/c2.txt"
	else:
		print("Invalid method")
		exit(1)
	iterations = 20

	sparkcontext = SparkContext("local", "KMeans clustering")
	data = sparkcontext.textFile(datasetfile)
	points = sparkcontext.parallelize(np.loadtxt(datasetfile))
	centroids = np.loadtxt(centroid_file)

	cost_values_list = []
	for index in range(iterations):
		points_with_centroids = points.map(assign_centroids(centroids))
		points_with_centroids.collect()

		cost_values = points_with_centroids.values().collect()
		cost_values = [[x for i, x in enumerate(a) if i != 0] for a in cost_values]
		cost_values_list.append(sum(sum(cost_values,[])))

		centroids_with_points = points_with_centroids.map(reverse_key_value)
		groupbykey = centroids_with_points.groupByKey()
		centroids = groupbykey.map(compute_new_centroids)
		centroids = centroids.sortByKey(ascending=True).values().collect()

	for index in range(len(centroids)):
		print("Centroid ", index)
		print(centroids[index])

	print("Cost Function values\n", cost_values_list)
	print("Percentage change in first 10 iterations ", (cost_values_list[0]-cost_values_list[9])*100/cost_values_list[0])
	fig = plt.figure()
	ax = plt.axes()
	ax.plot(range(0,20), cost_values_list)
	ax.set(title='Cost vs Iteration Method '+centroid_method, xlabel='Iteration', ylabel='Cost')
	plt.xticks(range(0, 20))
	plt.yticks(np.arange(100000000, 700000000, step=100000000))
	plt.savefig("CostvsIteration"+centroid_method+".png")
	sparkcontext.stop()

if __name__ == '__main__':
	main()
