import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def construieste_pipeline(model_name="LinearSVC", ngram_range=(1, 1), max_features=None):
    """
    Construiește un flux de procesare (Pipeline) scikit-learn standard pentru NLP.
    
    Fluxul este format din două etape:
    1. Vectorizarea TF-IDF: mapează documentul text pe un vector numeric, favorizând termenii
       care apar des în document, dar rar în întreaga colecție. Parametrul `sublinear_tf=True`
       aplică o transformare logaritmică pentru a diminua dominația termenilor extrem de frecvenți.
    2. Clasificatorul: aplică modelul de Machine Learning selectat pe vectorii obținuți.
    """
    if model_name == "LinearSVC":
        clf = LinearSVC(max_iter=2000)
    elif model_name == "Naive Bayes":
        clf = MultinomialNB()
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Model necunoscut")

    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            sublinear_tf=True 
        )),
        ('clf', clf)
    ])

def evalueaza_nlp(X_train, y_train, X_test, y_test, model_name="LinearSVC", ngram_range=(1,1)):
    """
    Antrenează pipeline-ul asamblat pe datele de train și evaluează performanța pe datele de test.
    Returnează modelul antrenat împreună cu metricile standard: Acuratețe, Matricea de Confuzie
    și Raportul de clasificare detaliat (Precizie, Recall, F1-Score).
    """
    timp_start = time.perf_counter()
    
    pipeline = construieste_pipeline(model_name, ngram_range)
    pipeline.fit(X_train, y_train)
    
    pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    
    timp_executie = time.perf_counter() - timp_start
    
    return pipeline, acc, cm, report, timp_executie