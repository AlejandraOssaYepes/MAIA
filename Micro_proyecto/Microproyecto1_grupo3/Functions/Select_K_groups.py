from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth 
from sklearn.metrics import  davies_bouldin_score, calinski_harabasz_score
import numpy as np 
import threading
import multiprocessing



class ClusterNumber:
    
    def __init__(self, image, n_bests) -> None:
        self.image = image 
        self.data_set = None 
        self.list_results_k = []
        self.K = None 
        self.best_algorithm = None 
        self.n_bests = n_bests 
        self.random_state = 12

    def ImageReshape(self): 
        self.data_set = self.image.reshape((-1, 3))


    def SearchN_MeanShiftAlgorithm(self):
        for q in np.arange(0.1, 0.6, 0.05):
            bandwidth = estimate_bandwidth(self.data_set, quantile=q, n_samples=100, random_state = self.random_state)
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs = -1)
            meanshift.fit(self.data_set)
            labels = meanshift.labels_
            centroids = meanshift.cluster_centers_
            n_clusters = centroids.shape[0]
            if n_clusters >= 5:
                davies_bouldin = davies_bouldin_score(self.data_set, labels)
                calinski_harabasz = calinski_harabasz_score(self.data_set, labels)
                self.list_results_k.append(('MeanShift',n_clusters, davies_bouldin,calinski_harabasz ))

            else: 
                pass 
        
    def SearchN_KmeansAlgorithm(self):

        for n_clusters in range(5,11):
            kmeans = KMeans(n_clusters=n_clusters, random_state = self.random_state)
            kmeans.fit(self.data_set)
            labels = kmeans.labels_
            davies_bouldin = davies_bouldin_score(self.data_set, labels)
            calinski_harabasz = calinski_harabasz_score(self.data_set, labels)
            self.list_results_k.append(('Kmeans',n_clusters, davies_bouldin,calinski_harabasz ))


    def Find_best_n(self):
        sort_list = sorted(self.list_results_k, key=lambda x: x[3], reverse=True)[:self.n_bests]
        sort_list = sorted(sort_list, key=lambda x: x[2], reverse=False)[0]
        self.K = sort_list[1]
        self.best_algorithm = sort_list[0]

    #Mejorar esto paralelizando 
        
    def Run(self):
        self.ImageReshape()
        self.SearchN_MeanShiftAlgorithm()
        self.SearchN_KmeansAlgorithm()

        self.Find_best_n()

        return self.best_algorithm ,self.K 

        




         













