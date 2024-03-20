import pandas as pd 
from nltk import RegexpTokenizer
import re 
from unidecode import unidecode
import spacy
from sklearn.model_selection import train_test_split

class PreprocessingClass:
    def __init__(self, rute, rute_bool, data) -> None:
        if rute_bool:
            self.data = pd.read_excel(rute)
            self.data.dropna(inplace=True)
            self.data.drop_duplicates(inplace=True)    
        
        else: 
            self.data = data
        self.nlp = spacy.load("es_core_news_sm")
            
       

    def read_data(self) -> pd.DataFrame :
        return self.data 
    
    def lower_text(self, text:str) -> str:
        return  text.lower()
    
    def remove_sepecial_characters(self, text: str) -> str:
        text  = unidecode(text)
        patron = r'[^a-zA-Z\s]'  
        clean_text = re.sub(patron, '', text)
        return  clean_text 

    def split_data(self):
        train, test = train_test_split(self.data, test_size=0.3, random_state=42)
        return train, test


    def remove_stopwords(self, text: str) -> str:
        stop_words = pd.read_csv('Data/Stopwords_spanish.csv', encoding='utf-8')
        stop_words.columns = ['id','word']
        stop_words_list = stop_words['word'].apply(lambda word: unidecode(word)).tolist()
        text_split = text.split()
        vector_nostopwords = [word for word in text_split if word not in stop_words_list ]
        return  ' '.join(vector_nostopwords) 

    def tokenization_text(self, text: str, overlap: bool, window_size: int) -> list:
        tokenizer = RegexpTokenizer(r'\w+')        
        tokens = tokenizer.tokenize(text)
        if overlap:
            tokens = [tokens[i:i+window_size] for i in range(len(tokens) - window_size + 1)]
            return  tokens 
        else: 
            return tokens

    def lemmatization_process(self, text: str) -> str:
        doc = self.nlp(text)
        emmas = [token.lemma_ for token in doc]
        return ' '.join(emmas)
    
    def remove_only_vocal_word(self, text: str) -> str:
        patron = r'\b[aeiouAEIOU]+\b'
        return re.sub(patron, '', text)
    
    def remove_references(self, text: str) -> str:
         text = re.sub(r'\([^)]*\)', '', text)
         text  = re.sub(r'\bww\w*\b', '', text)
         text = re.sub(r'\bx|x[^aeiouv\s]\w*\b', '', text)
         text = re.sub(r'\b\w*(ht|pd|url|html|uri|hr|mw|fer|cti|mt)\w*\b','' ,text)
         text = re.sub(r"\b\w+\.\w+\b", '', text)
         text = re.sub(r'\bv[^aeiou\s]\w*\b', '', text)
         return  text


    def Pipeline(self, text: str, overlap: bool, window_size: int) -> list:
       
        text = self.lower_text(text)
        text = self.remove_references(text)
        text = self.remove_only_vocal_word(text)
        text = self.remove_sepecial_characters(text)
        text = self.remove_stopwords(text)
        text = self.lemmatization_process(text)
        #remove sopt word generated by lemmatization
        text = self.remove_sepecial_characters(text)
        text = self.remove_stopwords(text)
        text = re.sub(r'\s+', ' ', text)
        tokens = self.tokenization_text(text, overlap, window_size)
        return tokens