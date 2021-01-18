#for dataframes
import pandas as pd
#for tokenize function
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

#for figure creation
import plotly.graph_objs as go
import plotly.express as px

#define tokenize function for later use
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

def return_figures(df):
    '''
    Args: df

    Returns: figures
    ––––––––––––––––––––––––––––––––––
    Creates plotly figures from df for plotting on dashboard

    Inludes: - a plotly.go bar graph
             - a plotly.xp distplot histrogam
             - a plotly.xp histogram
    '''
    # (long way of) sorting category values for plotting

    cat_df = df.iloc[:,4:].T.reset_index()
    cat_df['totals'] = cat_df.sum(axis=1)
    cat_df = cat_df.sort_values(by='totals', ascending=False)
    cat_df = cat_df[['index','totals']].reset_index(drop=True)
    cat_df.columns = ['Category', 'Total']
    cat_df = cat_df.T

    category_names = cat_df.iloc[0,:].values
    category_counts = cat_df.iloc[1,:].values
    #First plot, bar graph of Distribution of Message Categories
    graph_one = {
            'data': [
                go.Bar(
                    x=category_names,
                    y=category_counts,
                    marker={'color': category_counts,
                            'colorscale': 'sunset'}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                    },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                },
                'showlegend': False,
                'margin':{'l':85,'r':20,'b':150,'t':75,'pad':4}
            }
        }


    #Second plot, histogram of Number of Categories per Message
    totals = df.iloc[:,4:].sum(axis=1)
    graph_two = px.histogram(x=totals,
                             nbins = 9,
                             labels={
                             'x':'Number of Categories',
                             'y':'Total Messages'
                             },
                             range_x = [0,18],
                             title = 'Number of Categories per Message'
                      ).update(layout=dict(title=dict(x=0.5)))


    #get each message and find length of each
    m_list = df['message'].unique().tolist()
    mlen_list = [len(tokenize(message)) for message in m_list]

    #Third plot, histogram of Number of Words per Message
    graph_three = px.histogram(x=mlen_list,
                               marginal="box",
                               range_x = [0,100],
                               title = 'Number of Words per Message',
                               labels={'x':'Number of Words',
                                       'y':'Total'
                                  },
                            ).update(layout=dict(title=dict(x=0.5)))

    figures = []
    figures.append(graph_one)
    figures.append(graph_two)
    figures.append(graph_three)
    return figures
