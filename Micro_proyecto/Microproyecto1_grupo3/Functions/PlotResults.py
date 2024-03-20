import matplotlib.pyplot as plt 
import numpy as np 
from collections import Counter
from matplotlib.colors import ListedColormap


def Plot_Segmentation(image, img_segmented, model, labels, centroids):
    total_datos = img_segmented.shape[0]*img_segmented.shape[1]
    if model[1] == True:
        name = f'Fuzzy  m = {model[2]}'
    else:
        name = f'Bisecting KMeans'


    colors  = centroids[labels]
    list_colors = list(colors) 
    list_colors_tuples = [tuple(color) for color in list_colors]
    frecuencia_numeros = Counter(list_colors_tuples)
    colores = list(frecuencia_numeros.keys())
    frecuencias = list(frecuencia_numeros.values())
    colores_array = np.array([np.array(color) for color in colores])
    colores_number = [f'{i + 1}' for i in range(len(colores))]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Imagen Original')
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title(f'Imagen Segmentada con {name}')
    plt.imshow(img_segmented.astype(np.uint8))
    plt.axis('off')
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    plt.bar(colores_number, frecuencias, color = colores_array)
    plt.xlabel('Numero de Color')
    plt.title('Frecuencia de Colores')
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    for i, frecuencia in enumerate(frecuencias):
        porcentaje = frecuencia / total_datos * 100
        plt.text(colores_number[i], frecuencia, f'{porcentaje:.2f}%', ha='center', va='bottom')
    plt.show()
    

def PlotDimentionalReductionSelector(transform_Data,  centroids, labels):
    colors  = centroids[labels]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(transform_Data[:,0], transform_Data[:,1], c = colors)
    plt.title('Restultados usando el canl R y G')
    plt.subplot(1, 3, 2)
    plt.scatter(transform_Data[:,0], transform_Data[:,2], c = colors)
    plt.title('Restultados usando el canl R y B')
    plt.subplot(1, 3, 3)
    plt.scatter(transform_Data[:,1], transform_Data[:,2], c = colors)
    plt.title('Restultados usando el canl G y B')
    plt.show()


def PlotDimentionalReductionMethods(data_embedded, data_pca, centroids, labels):
    colors  = centroids[labels]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data_pca[:,0], data_pca[:,1], c = colors)
    plt.title('Resultados PCA')
    plt.subplot(1, 2, 2)
    plt.scatter(data_embedded[:,0], data_embedded[:,1], c = colors)
    plt.title('Restultados usando TSNE')
    plt.show()


def PlotPalet(centroids):
    cmap = ListedColormap(centroids)
    plt.figure(figsize=(12, 2))
    cb = plt.colorbar(plt.imshow(np.arange(len(centroids)).reshape(1, -1), cmap=cmap, aspect='auto'))
    plt.show()



