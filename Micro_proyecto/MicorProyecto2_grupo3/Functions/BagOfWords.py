from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import pandas as pd 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud


class BagOfWordsTfIdf:
    def __init__(self, data: pd.DataFrame, n_grams: tuple) -> None:
        self.data = data
        self.n_grams = n_grams
        self.documents  = self.data['Tokens'].apply(lambda x: ' '.join(x)).tolist()
        self.matriz_tfidf =  None
        self.vectorizador_tfidf = None 
            
    def vectorization(self) -> np.array:
        self.vectorizador_tfidf  = TfidfVectorizer(ngram_range=self.n_grams)
        self.matriz_tfidf  = self.vectorizador_tfidf .fit_transform(self.documents)
        return self.matriz_tfidf
    
    def topics_analisis(self, n_components) -> tuple:
        lda = LatentDirichletAllocation(n_components = n_components,  random_state=123,  learning_method='batch')
        print(self.matriz_tfidf.shape)
        X_topics = lda.fit_transform(self.matriz_tfidf)
        return X_topics, lda 
    
    def CloudWords(self, n_top_words, lda):
        feature_names = self.vectorizador_tfidf.get_feature_names_out()
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))

        # Ajustar el espacio entre subplots
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.01)

        for topic_idx, topic in enumerate(lda.components_):
            wc_text = ' '.join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]])
            wordcloud = WordCloud(
            width=800,
            height=700,
            background_color ='white',
            min_font_size=20,
            max_font_size=110).generate(wc_text)
            ax = axs[topic_idx // 4, topic_idx % 4]
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topico {topic_idx + 1}')
            ax.axis('off')
        plt.show() 
        return True




        


        

