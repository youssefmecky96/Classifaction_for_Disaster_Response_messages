import sys
import nltk
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re 
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
     """
    usuage loads data from a database 
    input: database_filepath
    output: X input to ML model Y result variavle and columns name
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table('project_4',engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    return X,Y,list(Y.columns)


def tokenize(text):
    """
    usuage normalizes case,removes punctuation,tokenizes words,lemmatize and removes stop words 
    input: text
    output: tokenized text
    """
     # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("english")]

    return tokens


def build_model():
    """
    usuage builds a pipline that tokenizes the input ,uses Tdidf transform to extract features and uses that as an input a Kneighbour classifier, performs grid search as well 
    input: none
    output: the ideal model from the grid search
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    
    ('clf',  MultiOutputClassifier(KNeighborsClassifier()))
    ])
    parameters = parameters = {
        'tfidf__use_idf': (True, False),
        
        'clf__estimator__weights':('uniform','distance')}

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs = -1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    usuage outputs scores for model  
    input: model , X_test test data , Y_test correct results, category_names
    output:prints scores for how well the model preforms
    """
    y_pred=model.predict(X_test)
    print(
        classification_report(Y_test, y_pred, target_names=category_names)
    )


def save_model(model, model_filepath):
    """
    usuage saves model into a pickle file
    input: model ,model_filepath
    output:no output
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
