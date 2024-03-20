Descripción de los Scripts.

1. Se ha construido una Pipeline que se ejecuta desde el notebook principal. En ella, se realiza una limpieza previa del DataFrame que incluye la eliminación de duplicados y de valores vacíos. Se han implementado varias funciones de procesamiento, como la transformación de los textos a minúsculas con lower_text(), la eliminación de caracteres especiales (números, tildes, ñ, etc.) con remove_special_characters(), y la eliminación de palabras identificadas como innecesarias y poco aportates para la sintaxis de los textos. Estas últimas incluyen palabras entre paréntesis que son citas bibliográficas, URLs, páginas web, y palabras con estructuras anormales conformadas solo por vocales. Esta eliminación se ha llevado a cabo en la función remove_references().

Además, se ha realizado la eliminación de palabras de parada que no aportan significado semántico al texto por sí mismas, implementada en la función remove_stopwords(). Finalmente, se ha realizado la lematización en lemmatization_process() con la libreria 
, que reduce las palabras a su forma raíz o lema eliminando prefijos, sufijos y flexiones gramaticales permitiendo comparar palabras con diferentes formas gramaticales; y la tokenización con tokenization_text(), que divide el texto en unidades más pequeñas (tokens) y cuenta con la opción de (overlap, window_size) ya que la tokenización con solapamiento permite capturar mejor el contexto y la información importante en un texto y generan tokens que se superponen parcialmente entre sí, lo que significa que una palabra puede estar representada en múltiples tokens.

La Ejecución se lleva acabo en la extensión Functions.Preprocessing import PreprocessingClass y que es implemenetado con la función preprocessor.Pipeline(text = text, overlap= overlap, window_size=window_size)

Dividimos el conjunto de datos en conjuntos de prueba y entrenamiento para evaluar el rendimiento de los modelos. Utilizamos el conjunto de datos completo para el análisis de tópicos, mientras que para la visualización y el primer modelo de clasificación solo utilizamos el conjunto de entrenamiento.

2. Bolsas de Palabras
Se implementa una "Bag of Words" (BoW) que es un modelo que se utiliza en el procesamiento del lenguaje natural (PLN) para representar documentos de texto, empleando la función TfidfVectorizer(ngram_range=self.n_grams) que extrae características de un conjunto de documentos y convierte cada documento en un vector de características, donde cada característica representa la importancia de una palabra o frase y se calcula utilizando la frecuencia de término-documento (TF) y la frecuencia inversa de documento (IDF). La vectorización vectorization(self) tiene como parámetros ngram_range que especifican el rango de n-gramas que es una secuencia de n elementos consecutivos de una cadena de texto, como palabras o caracteres y se define como una tupla (min_n, max_n) que especifica el rango de n-gramas a considerar.

Para el análisis de tópicos se utiliza el modelo de Asignación Latente de Dirichlet (LDA), que es un modelo estadístico generativo. Este modelo asume que cada documento está representado por una distribución latente de tópicos y que cada tópico está representado por una distribución latente de palabras. La función topics_analysis() implementa este modelo utilizando la librería LatentDirichletAllocation para identificar estos tópicos. Se utilizan parámetros como n_components para especificar el número de tópicos que se deben extraer del corpus. En este caso, se conocen por el contexto de los textos que son 16 Objetivos de Desarrollo Sostenible (ODS) que corresponden a las etiquetas con las que cada texto está relacionado.

La Ejecución se lleva acabo en la extensión Functions.BagOfWords import BagOfWordsTfIdf

3. Reducción de Dimensionalidad
Para reducir la dimensionalidad de los datos antes de pasarlos al clasificador, hemos utilizado la Descomposición en Valores Singulares (SVD) implementada en la librería SciPy y ejecutado con la función DimentionalReduction(). Con SVD, proyectamos los datos originales en un espacio de 100 dimensiones. Además, como ejercicio visual, hemos graficado los datos en un espacio de 2 dimensiones. Esta representación gráfica en 2D nos permite visualizar la distribución de los datos y entender mejor cómo se agrupan o dispersan en función de las características seleccionadas en la reducción de dimensionalidad. Se hace usu de la funcion TSNE() para reducir su dimensionalidad y permitir la visualización de los datos en un plano.

La Ejecución se lleva acabo en la extensión Functions.ClasificationTfIdf import ClassificationProcees

4. Busqueda de Hiperparametros
Se implementa RandomizedSearchCV() que es una función de la biblioteca scikit-learn utilizada para realizar una búsqueda aleatoria de hiperparámetros encontrando la mejor configuración para un modelo sin tener que probar manualmente todas las combinaciones posibles. Este desarrollo se realizó en la función clasificator.SearchModel() con los siguientes rangos de configuraciones:

'max_depth': randint(1, 50)
'n_estimators': randint(10, 500)
'criterion': ["gini", "entropy", "log_loss"]
Debido al desbalanceo de clases de utilizo el scoring='f1_macro que se calcula como la media ponderada de las precisiones y las revocaciones de cada clase; además se implementó un cv=5 que especifica una validación cruzada de 5 pliegues para evaluar cada combinación de hiperparámetros y n_iter=40 que indica que se probarán 40 combinaciones aleatorias de hiperparámetros.

De forma que sea comparable los resultados del modelo se establecion un random_state=42

La Ejecución se lleva acabo en la extensión Functions.ClasificationTfIdf import ClassificationProcees

5. Predicción
El modelo de RandomForestClassifier es un clasificador de ensamble que se basa en la técnica de Bosques Aleatorios (Random Forests), este modelo combina múltiples árboles de decisión durante el entrenamiento y devuelve la clase más frecuente como la predicción final. Cada árbol de decisión en el bosque se entrena con una submuestra aleatoria del conjunto de datos y una selección aleatoria de características ayudando a reducir el sobreajuste y a mejorar la generalización del modelo; esta implementacion se realiza con el **best_params etregado por el paso anterior, el cual es un diccionario que contiene los valores de hiperparámetros que se determinó que eran óptimos para el modelo. Su ejecución se realiza con la funcion Predict_labels() aplicando la libreria RandomForestClassifier(**best_params, n_jobs = -1, random_state=42) de sklearn.

La Ejecución se lleva acabo en la extensión Functions.PredictionTdldf import PredictProcess

6. Mapas autoorganizados
Los mapas autoorganizados (SOM) son una herramienta poderosa para visualizar conjuntos de datos de alta dimensionalidad en un espacio bidimensional, preservando las relaciones topológicas entre los datos. Son útiles para identificar patrones, grupos y relaciones en los datos. En el desarrollo del notebook se presenta la representación visual de esta herramienta en la función PlotSom() con ayuda de la librería 
 con las siguiente configuración MiniSom(10, 10, data.shape[1], sigma=1.5, learning_rate=.7, activación_distance='euclidean', topology='hexagonal', vecindad_function='gaussian', random_seed=10) en donde se entrena con 100 neuronas y una topología 'hexagonal'

La Ejecución se lleva acabo en la extensión Functions.SOM import PlotSom
