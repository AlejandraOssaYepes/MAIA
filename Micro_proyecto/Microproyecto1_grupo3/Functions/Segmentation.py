from sklearn.cluster import  BisectingKMeans
import skfuzzy as fuzz
import numpy as np 
from sklearn.metrics import  davies_bouldin_score



class SegmentationProcess:

    def __init__(self, K, image) -> None:
        self.K = K 
        self.data_set = None 
        self.image = image 
        self.random_state = 12

    def ImageReshape(self): 
        self.data_set = self.image.reshape((-1, 3))


    def Segmentation_BisectingKMeans(self):
        algorithm = BisectingKMeans(n_clusters= self.K, random_state=self.random_state )
        algorithm.fit(self.data_set)
        labels = algorithm.labels_ 
        centroids = algorithm.cluster_centers_
        img_segmented = centroids[labels]*255
        img_segmented = img_segmented.reshape(self.image.shape)  
        return centroids,  img_segmented, labels


    def Segmentation_FuzzyCmeans(self, m):
        cntr, u, u0, d, jm, p, fpc= fuzz.cluster.cmeans(self.data_set.T, self.K, m, 0.005,  1000 )
        labels = np.argmax(u, axis=0)
        centroids = cntr 
        img_segmented = centroids[labels]*255
        img_segmented = img_segmented.reshape(self.image.shape)   
        return  centroids,  img_segmented, labels
    
    def Metrics(self, fuzzy=False, m = None):
        if fuzzy:
            centroids,  img_segmented, labels = self.Segmentation_FuzzyCmeans(m)
        else:
            centroids,  img_segmented, labels = self.Segmentation_BisectingKMeans()

        davies_bouldin = davies_bouldin_score(self.data_set, labels)
        return centroids,  img_segmented, labels, davies_bouldin 
    
    def Run(self):
        self.ImageReshape()
        models =  [('BisectingKMeans', False, None), ('Fuzzy-1.1', True, 1.1), 
                   ('Fuzzy-1.5', True, 1.5), ('Fuzzy-2.5', True, 2.5)]
        
        list_results = []
        for model in models:
            centroids,  img_segmented, labels, davies_bouldin  =  self.Metrics(fuzzy=model[1], m = model[2])
            list_results.append((centroids,  img_segmented, labels, davies_bouldin , model))

        sort_list = sorted(list_results, key=lambda x: x[3])
        return sort_list[0]
      
    
 
