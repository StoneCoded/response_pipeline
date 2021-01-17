import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier



def load_data(database_filepath):
    '''
    Args: database_filepath

    Returns: X, y, y.columns
    ––––––––––––––––––––––––––––––––––
    Loads dataframe from database and returns X, y and category values for
    train_test_split and evaluate_model functions

    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    myQuery = '''SELECT * FROM DisasterResponse'''
    df = pd.read_sql(myQuery, engine)
    X = df.message.values
    y = df.iloc[:,4:]
    return X, y, y.columns

def tokenize(text, stop = True):
    '''
    Args: text, stop(default: True)

    Returns: clean_tokens
    ––––––––––––––––––––––––––––––––––
    Takes text input and tokenizes the string,
    Removes stop words if stop = True,
    Stems and lemmatizes text,
    Lowers the case and removes punctuation.
    '''
    tokens = word_tokenize(text)
    #Remove Stop Words
    if stop == True:
        tokens = [w for w in tokens if w not in stopwords.words("english")]
    #Stem, lem, lower and strip
    clean_tokens = []
    for tok in tokens:
        tok = PorterStemmer().stem(tok)
        tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        clean_tokens.append(tok)
    return tokens

def build_model():
    # pipeline = Pipeline([('vect',CountVectorizer(tokenizer = tokenize)),
    #                      ('tfidf',TfidfTransformer()),
    #                      ('clf', MultiOutputClassifier(RandomForestClassifier()
    #                         ))
    #                     ])
    # parameters ={
    #             'clf__estimator__n_estimators': [30, 60],
    #             'clf__estimator__min_samples_split': [2, 4]
    #              }
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # small set of parameters because of time
    parameters = {
        'vect__max_df':[0.75,1.0],
        'clf__estimator__n_estimators': [20, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    # return pipeline
    # train classifier
    # cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    for x, col in enumerate(labels):
        print(col)
        print(classification_report(y_test[col], y_pred[:,x]))
    accuracy = (y_pred == y_test.values).mean()
    print(accuracy)
    print(model.best_estimator_)

def save_model(model, model_filepath):
    '''save as pickle file'''
    joblib.dump(model, model_filepath, compress=3)
    # pickle.dump(model, open(model_filepath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
                                                            test_size=0.2, \
                                                            random_state = 42)
        print('Building model...')
        model = build_model( )

        print('Training model...')
        model.fit(X_train, Y_train)
        #
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        #
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        #
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
