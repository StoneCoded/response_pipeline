import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath = 'data/disaster_messages.csv', categories_filepath='data/disaster_categories.csv'):
    '''
    Args: messages_filepath, categories_filepath

    Returns: df
    ––––––––––––––––––––––––––––––––––
    Creates a dataframe from disaster_messages.csv and disaster_categories.csv
    by merging them together
    Has a default filepath of data folder where originals are stored.

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# def taxonomy(df):
#     '''
#     Args: def
#
#     Returns: df
#     ––––––––––––––––––––––––––––––
#     Takes in a dataframe and groups some of the categories together.
#     Intended for specific use, not general.
#
#     '''
#     df = df.copy()
#     df['Human_Affected'] = df[['missing_people','child_alone', 'death',\
#                                        'refugees']].sum(axis=1)
#     df['General_Aid'] = df[['aid_related','other_aid', \
#                                     'aid_centers']].sum(axis=1)
#     df['Medical'] = df[['medical_help','medical_products', \
#                                 'hospitals']].sum(axis=1)
#     df['Human_Aid'] = df[['military','security', \
#                                   'search_and_rescue']].sum(axis=1)
#     df['Basic_Needs'] = df[['electricity','food', 'money', 'tools', \
#                                     'clothing', 'shelter']].sum(axis=1)
#     df['Infrastructure'] = df[['buildings','infrastructure_related', \
#                                'other_infrastructure', 'shops', 'transport'] \
#                                ].sum(axis=1)
#     df['Area_Conditions'] = df[['weather_related','floods', 'storm', 'fire',\
#                            'earthquake', 'cold', 'other_weather']].sum(axis=1)

    #Ungrouped categories 'related', 'request', 'offer', 'water', 'direct_report'
    #as they are either ambiguous(water(could be a need or rain/flood)),
    #fine as is or unclear.
    #Definitely room to improve model at this point.


    # df = df.drop(['missing_people','child_alone', 'death', 'refugees',\
    # 'aid_related','other_aid', 'aid_centers','medical_help','medical_products',\
    # 'hospitals','military','security', 'search_and_rescue','electricity','food',
    # 'money','tools', 'clothing', 'shelter', 'buildings',\
    # 'infrastructure_related', 'other_infrastructure', 'shops', 'transport',\
    # 'weather_related','floods','storm', 'fire', 'earthquake', 'cold', \
    # 'other_weather'], axis=1)

    # for col in df:
    #     df[col] = df[col].apply(lambda x: 1 if x >= 1 else 0)
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    return df
def clean_data(df):
    '''
    Args: df
    Returns: df
    ––––––––––––––––––––––––––––––––––
    Takes df and cleans it by:
    1. Splitting categories into individual columns
    2. Converts category values to just numeric values
    3. Replace `categories` column in `df` with new category columns.
    4. Drop genre column and drop child_alone column as all are 0
    5. Remove duplicates
    6. Remove 'related' column value of 2
    '''
    #1.
    categories = df['categories'].str.split(';', expand=True).copy()
    categories.columns = (categories.iloc[0,:].str.slice(stop=-2).tolist())
    #2.
    categories = categories.apply(lambda x: pd.to_numeric(x.str.slice(start=-1)))

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # categories = categories[categories.related != 2] (for taxonomy)
    # categories = taxonomy(categories) #Needs work before deploying
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    #3.
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    #4.
    df = df.drop(['genre','child_alone'], axis = 1)
    #5.
    df = df.drop_duplicates()
    df = df.dropna().reset_index(drop=True)
    #6.
    df = df[df['related']!=2]
    #From figure eights website, a value of 2 means no (or potenitally 0),
    #All other columns are also null
    #As odd group is relatively small I've chosen to remove values

    return df



def save_data(df, database_filename):
    '''
    Args: df, database_filename
    Returns: none
    ––––––––––––––––––––––––––––––––––
    Saves dataframe to database
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning up this dirty data...')
        df = clean_data(df)

        print('Saving the data to {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
