import sys
#for data manipulation
import pandas as pd
import joblib
from sqlalchemy import create_engine
#for natural language processing
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
#for model
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

class StartingVerbExtractor():
    """
    Creates Verb Extractor class

    Get the starting verb of each sentence making a new feature
    for the classifier later on.
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

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
        tokens = [w for w in tokens if w not in stopwords.words('english')]
    #Stem, lem, lower and strip
    clean_tokens = []
    for tok in tokens:
        # tok = PorterStemmer().stem(tok) I'm not allowed to stem :(
        tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        clean_tokens.append(tok)
    return tokens

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
# pipeline.get_params()


    # small set of parameters because of time
    parameters = {'clf__estimator__n_estimators' : [60, 80]}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return cv

def evaluate_model(model, X_test, y_test, labels):
    '''
    evaluates models accuracy by producing a classification report for each
    category along with an overall accuracy

    *for added fun it prints the best estimator from the GridSearchCV*
    '''
    y_pred = model.predict(X_test)
    for x, col in enumerate(labels):
        print(col)
        print(classification_report(y_test[col], y_pred[:,x]))
    accuracy = (y_pred == y_test.values).mean()
    print(accuracy)
    print(model.best_estimator_)

def save_model(model, model_filepath):
    '''
    Args: model, model_filepath

    Returns: none
    ––––––––––––––––––––––––––––––––––
    Saves model using joblib and compresses slightly for a smaller filesize
    '''
    joblib.dump(model, model_filepath, compress=3)

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
