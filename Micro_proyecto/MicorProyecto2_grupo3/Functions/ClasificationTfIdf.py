from numpy.core.multiarray import array as array
from Functions.Preprocessing import PreprocessingClass
from scipy.sparse.linalg import svds
from scipy.stats import randint
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix



class ClassificationProcees():
    
    def __init__(self, data, vectorizer) -> None:
        self.preprocessor = PreprocessingClass(rute=None, rute_bool=False, data=data )
        self.data = data
        self.data['Tokens'] = self.data['textos'].swifter.apply(lambda text: self.preprocessor.Pipeline(text = text, overlap= False, window_size=2) )
        self.label = data['ODS']
        self.documents  = self.data['Tokens'].apply(lambda x: ' '.join(x)).tolist()
        self.X = vectorizer.transform(self.documents)
        
    def DimentionalReduction(self):
        u, s, vt = svds(self.X, k=100)
        self.vt = vt
        proyection = self.X.dot(vt.T)
        return proyection, vt
    
    
    def SearchModel(self, proyeccion):
        x_train  = proyeccion 
        label_aux = csr_matrix(self.label.values.reshape(-1, 1))
        y_train = label_aux.toarray().reshape(-1)

        model = RandomForestClassifier(n_jobs = -1, random_state = 42)
        param_dist = {
        'max_depth': randint(1, 50),  
        'n_estimators': randint(10, 500),  
        'criterion': ["gini", "entropy", "log_loss"]
        }
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=40, cv=5, scoring='f1_macro', random_state=42, verbose=2, n_jobs=-1)
        random_search.fit(x_train, y_train)
        
        best_params = random_search.best_params_
        best_score = random_search.best_score_
   
        return best_params, best_score, x_train, y_train









    
    



        
    

    

    