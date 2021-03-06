#for data
import json
import plotly
import pandas as pd
from sqlalchemy import create_engine
import joblib

#for tokenize function
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import wordnet
nltk.download(['punkt', 'wordnet', 'stopwords'])

#for web app
from flask import Flask
from flask import render_template, request, jsonify

from figures import return_figures

from sklearn.base import BaseEstimator, TransformerMixin
app = Flask(__name__)
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
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
# load data
engine = create_engine('sqlite:///../data/ResponseData.db')
myQuery = '''SELECT * FROM DisasterResponse'''
df = pd.read_sql(myQuery, engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Args: None
    Returns render_template
    ––––––––––––––––––––––––––––––––––
    Gets plotly figures from return_figures and preps
    them and sends to web page
    '''
    graphs = return_figures(df)
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    '''
    Args: None
    Returns render_template
    ––––––––––––––––––––––––––––––––––
    Uses data model to predict classification of query and sends to webpage
    '''
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template('go.html',
                            query=query,
                            classification_result=classification_results
                        )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
