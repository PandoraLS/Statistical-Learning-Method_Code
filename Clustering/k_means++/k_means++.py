# -*- coding: utf-8 -*-
# @Time    : 2021/2/18 10:34
# @Author  : sen

"""
k-means++相关知识参考[百面机器学习]
参考连接：
http://www.kazemjahanbakhsh.com/codes/k-means.html
https://github.com/kjahan/k_means
"""

import pandas as pd
import random as rand
import math
import matplotlib.pyplot as plt
import csv
import os

if os.path.exists("k_means.log"):
    os.remove("k_means.log")
# import logger # 如果不需要logger把这行注释掉就ok了


class Point:
    def __init__(self, latit_, longit_):
        self.latit = latit_ # 纬度
        self.longit = longit_ # 经度

    def euclidean_distance(self, another_point):
        """
        计算当前节点与另一个节点的欧氏距离
        :param another_point: 另一个节点
        :return: 欧式距离
        """
        return math.sqrt(math.pow(self.latit - another_point.latit, 2.0) + math.pow(self.longit - another_point.longit, 2.0))


class KMeans:
    def __init__(self, geo_locs_, k_):
        self.geo_locations = geo_locs_ # location
        self.k = k_ # k值
        self.clusters = None  # clusters of nodes
        self.means = []     # means of clusters
        self.debug = False  # debug flag

    def next_random(self, index, points, clusters):
        """
        随机返回下一个随机node
        选择与其他节点的最大距离的下一个节点
        # this method returns the next random node
        # pick next node that has the maximum distance from other nodes
        :param index:
        :param points: 数据集中所有节点
        :param clusters: 节点集群
        :return:
        """
        dist = {}
        for point_1 in points:
            if self.debug:
                print("point_1: {} {}".format(point_1.latit, point_1.longit))

            # 计算这个节点与集群中所有其他点的距离
            for cluster in clusters.values():
                point_2 = cluster[0]
                if self.debug:
                    print("point_2: {} {}".format(point_2.latit, point_2.longit))
                if point_1 not in dist:
                    dist[point_1] = math.sqrt(math.pow(point_1.latit - point_2.latit,2.0) + math.pow(point_1.longit - point_2.longit,2.0))
                else:
                    dist[point_1] += math.sqrt(math.pow(point_1.latit - point_2.latit,2.0) + math.pow(point_1.longit - point_2.longit,2.0))
        if self.debug:
            for key, value in dist.items():
                print("({}, {}) ==> {}".format(key.latit, key.longit, value))
        # 现在让我们返回与前面节点的最大距离的点
        count_ = 0 # 统计dict的数目
        max_distance = 0 # 距离最大值
        for key, value in dist.items():
            if count_ == 0:
                max_distance = value
                max_point = key
                count_ += 1
            else:
                if value > max_distance:
                    max_distance = value
                    max_point = key
        return max_point

    def initial_means(self, points):
        """
        计算初始means，随机选择第一个节点
        :param points: 所有节点
        :return:
        """
        point_ = rand.choice(points)
        if self.debug:
            print("point#0: {} {}".format(point_.latit, point_.longit))
        clusters = dict()
        clusters.setdefault(0, []).append(point_)
        points.remove(point_)

        # 现在我们选择剩下的 k-1 个点
        for i in range(1, self.k):
            point_ = self.next_random(i, points, clusters)
            if self.debug:
                print("point#{}: {} {}".format(i, point_.latit, point_.longit))
            #clusters.append([point_])
            clusters.setdefault(i, []).append(point_)
            points.remove(point_)
        # compute mean of clusters
        self.means = self.compute_means(clusters)
        if self.debug:
            print("initial means:")
            self.print_means(self.means)

    def compute_means(self, clusters):
        """
        计算集群的中心点
        :param clusters: 集群
        :return:
        """
        means = []
        for cluster in clusters.values():
            mean_point = Point(0.0, 0.0)
            cnt = 0.0
            for point in cluster:
                #print "compute: point(%f,%f)" % (point.latit, point.longit)
                mean_point.latit += point.latit
                mean_point.longit += point.longit
                cnt += 1.0
            mean_point.latit = mean_point.latit/cnt
            mean_point.longit = mean_point.longit/cnt
            means.append(mean_point)
        return means

    def assign_points(self, points):
        """
        将节点分配到距离最小的集群中
        :param points: 点 集
        :return:
        """
        if self.debug:
            print("assign points")
        clusters = dict()
        for point in points:
            dist = []
            if self.debug:
                print("point({},{})".format(point.latit, point.longit))
            # find the best cluster for this node
            for mean in self.means:
                dist.append(math.sqrt(math.pow(point.latit - mean.latit,2.0) + math.pow(point.longit - mean.longit,2.0)))
            # let's find the smallest mean
            if self.debug:
                print(dist)
            cnt_ = 0
            index = 0
            min_distance = dist[0]
            for d in dist:
                if d < min_distance:
                    min_distance = d
                    index = cnt_
                cnt_ += 1
            if self.debug:
                print("index: {}".format(index))
            clusters.setdefault(index, []).append(point)
        return clusters

    def update_means(self, means, threshold):
        # compare current means with the previous ones to see if we have to stop
        for i in range(len(self.means)):
            mean_1 = self.means[i]
            mean_2 = means[i]
            if self.debug:
                print("mean_1({},{})".format(mean_1.latit, mean_1.longit))
                print("mean_2({},{})".format(mean_2.latit, mean_2.longit))
            if math.sqrt(math.pow(mean_1.latit - mean_2.latit,2.0) + math.pow(mean_1.longit - mean_2.longit,2.0)) > threshold:
                return False
        return True

    def save(self, filename="output.csv"):
        # save clusters into a csv file
        with open(filename, mode='w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['latitude', 'longitude', 'cluster_id'])
            cluster_id = 0
            for cluster in self.clusters.values():
                for point in cluster:
                    writer.writerow([point.latit, point.longit, cluster_id])
                cluster_id += 1

    def print_clusters(self, clusters=None):
        # debug function: print cluster points
        if not clusters:
            clusters = self.clusters
        cluster_id = 0
        for cluster in clusters.values():
            print("nodes in cluster #{}".format(cluster_id))
            cluster_id += 1
            for point in cluster:
                print("point({},{})".format(point.latit, point.longit))

    def print_means(self, means):
        # debug function: print means
        for point in means:
            print("{} {}".format(point.latit, point.longit))

    def fit(self, plot_flag):
        # Run k_means algorithm
        if len(self.geo_locations) < self.k:
            return -1   #error
        points_ = [point for point in self.geo_locations]
        # compute the initial means
        self.initial_means(points_)
        stop = False
        iterations = 1
        print("Starting K-Means...")
        while not stop:
            # assignment step: assign each node to the cluster with the closest mean
            points_ = [point for point in self.geo_locations]
            clusters = self.assign_points(points_)
            if self.debug:
                self.print_clusters(clusters)
            means = self.compute_means(clusters)
            if self.debug:
                print("means:")
                self.print_means(means)
                print("update mean:")
            stop = self.update_means(means, 0.01)
            if not stop:
                self.means = []
                self.means = means
            iterations += 1
        print("K-Means is completed in {} iterations. Check outputs.csv for clustering results!".format(iterations))
        self.clusters = clusters
        #plot cluster for evluation
        if plot_flag:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            markers = ['o', 'd', 'x', 'h', 'H', 7, 4, 5, 6, '8', 'p', ',', '+', '.', 's', '*', 3, 0, 1, 2]
            colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', [0,0,1]]
            cnt = 0
            for cluster in clusters.values():
                latits = []
                longits = []
                for point in cluster:
                    latits.append(point.latit)
                    longits.append(point.longit)
                ax.scatter(longits, latits, s=60, c=colors[cnt], marker=markers[cnt])
                cnt += 1
            plt.savefig('result.png')
            plt.show()
        return 0

def main(dataset_fn, output_fn, clusters_no):
    """
    :param dataset_fn (csv): 数据集
    :param output_fn (csv): 纬度、经度、分类的id
    :param clusters_no: 分多少个类别
    :return:
    """
    geo_locs = [] # 保存Point(纬度, 经度)
    # read location data from csv file and store each location as a Point(latit,longit) object
    df = pd.read_csv(dataset_fn)
    for index, row in df.iterrows():
        loc_ = Point(float(row['LAT']), float(row['LON']))  # tuples for location
        geo_locs.append(loc_)
    # run k_means clustering
    model = KMeans(geo_locs, clusters_no)
    flag = model.fit(True)
    if flag == -1:
        print("No of points are less than cluster number!")
    else:
        # save clustering results is a list of lists where each list represents one cluster
        model.save(output_fn)

if __name__ == '__main__':
    input_fn = "NYC_Free_Public_WiFi_03292017.csv"
    output_fn = 'output.csv'
    main(dataset_fn=input_fn, output_fn=output_fn, clusters_no=8)