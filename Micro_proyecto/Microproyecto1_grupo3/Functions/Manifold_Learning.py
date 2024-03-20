from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class DimentionalRedcution:
    def __init__(self, image, centroids, labels) -> None:
        self.image = image 
        self.centroids = centroids
        self.labels = labels 
        self.data_set = None 
        
        
    def Transform_Data(self):
        self.data_set = self.image.reshape((-1, 3))

    def Dimentional_reduction_Algorithms(self):
        self.Transform_Data()
        data_embedded = TSNE(n_components=2, learning_rate='auto',
                                  init='random', perplexity=20, n_jobs = -1).fit_transform(self.data_set)
        data_pca = PCA(n_components=2).fit_transform(self.data_set)

        return data_embedded,  self.data_set, data_pca


         