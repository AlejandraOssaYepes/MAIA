from Functions.Preprocessing import PreprocessingClass
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

class PredictProcess():

    def __init__(self, data, vectorizer) -> None:
        self.preprocessor = PreprocessingClass(rute=None, rute_bool=False, data=data )
        self.data = data
        self.data['Tokens'] = self.data['textos'].swifter.apply(lambda text: self.preprocessor.Pipeline(text = text, overlap= False, window_size=2) )
        self.label = data['ODS']
        self.documents  = self.data['Tokens'].apply(lambda x: ' '.join(x)).tolist()
        self.X = vectorizer.transform(self.documents)

    def Predict_labels(self, vt, best_params, x_train, y_train):
        x_test  = self.X.dot(vt.T)
        label_aux = csr_matrix(self.label.values.reshape(-1, 1))
        y_test = label_aux.toarray().reshape(-1)
        model = RandomForestClassifier(**best_params, n_jobs = -1, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_test, y_pred
    
    def metrics_calculate(self, y_test, y_pred ):
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        f1macro = f1_score(y_test, y_pred, average= 'macro') 
        return accuracy, class_report, f1macro
