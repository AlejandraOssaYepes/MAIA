Descripción de los Scripts.

Read_images: En este script se realiza el pipeline de procesamiento de las imágenes, donde se llevan a cabo las siguientes etapas: se lee la imagen, se reduce su tamaño utilizando interpolación INTER_AREA, se suaviza la imagen y se normaliza. Se crea una clase para el pipeline y se inicializa con el método Pipeline.

Select_K_groups: En este script se escoge el valor óptimo de K para los clusters. Para ello, se utiliza el algoritmo de K-Means y MeanShift variando sus parámetros y comparando con los coeficientes de Davies-Bouldin y Calinski-Harabasz. Se crea una clase para este proceso y se ejecuta en el método Run.

Segmentation: En este caso se utilizan el valor óptimo de K con el algoritmo de Fuzzy C-Means y Bisecting K-Means, para escoger la mejor agrupación de acuerdo a la métrica de davies_bouldin_score. Para ello se implementó una clase y se llama desde el método Run.

Manifold_Learning: Este script implementa el método de t-SNE para reducir la dimensión, utilizando el parámetro perplexity en 20. Se creó una clase para este proceso y se utiliza desde el método Tsne_Algorithm. Adicionalmente se usa el método de PCA par ala reudcción de dimensión

PlotResults: En este script se escribieron las funciones para plotear las imágenes y el diagrama de barras adicionales de los resultados de la reducción de dimensión y la paleta de colores.



