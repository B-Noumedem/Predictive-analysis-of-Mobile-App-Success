import os

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit



def contentAdvisoryRating_recode(val):
    if val == '4_plus':
        output = 1
    elif val == '9_plus':
        output = 2
    elif val == '12_plus':
        output = 3
    else:
        output = 4
    return output

def date_diff(d1, d2):
    """Returns the difference between two dates in months"""
    d1 = datetime.strptime(d1[:10], "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return round(abs((d2 - d1)).days*(12/365),2)

def get_month(d1):
    """Return release month of application """
    return datetime.strptime(d1[:10], "%Y-%m-%d").strftime('%m-%b')

def search_languages(data):
    unique_lang = []
    for list_lang in data.languageCodesISO2A:
        for lang in list_lang:
            if unique_lang.count(lang)==0:
                unique_lang.append(lang)
    return unique_lang  

def dummy_langue(sub_liste,liste):
    '''Function that creates as many dummy variables as there are languages available for the application.
    
    Input:
        sub_liste: list of languages availaible for the application i-e
                   a cell of the column 'languageCodesISO2A'
        liste: list of availaible languages'''
    
    results=[]
    for elt in liste:
        y = ( sub_liste.count(elt) != 0 ) * 1 # test if the current elt is in the  subliste
        results.append(y)
        
    return results

def create_lang_df(data):
    unique_lang = search_languages(data)
    data_lang = data.languageCodesISO2A.apply(lambda x:dummy_langue(x, unique_lang))
    data_lang = data_lang.apply(pd.Series)
    data_lang = data_lang.rename(columns=dict(zip(range(132),unique_lang))) 
    data_lang.index = data.id_app

    return data_lang

def search_languages_top(data_lang, nb):
    cum_lang = data_lang.sum(axis=0).sort_values(ascending=False)
    top_lang = list(cum_lang[0:10].index)
    return top_lang

def search_genres(data):
    unique_genre = []
    for list_genre in data.genres:
        for genre in list_genre:
            if unique_genre.count(genre)==0:
                unique_genre.append(genre)  
    return unique_genre  

def dummy_genre(sub_liste,liste):
    '''Function that creates as many dummy genre as there are genre available for the application.
    
    
    Input:
        sub_liste: all the genres of one application, 
                   a a cell of the column 'Genres'
        liste: extended definition of the variable Genres'''
    
    results=[]
    for elt in liste:
        y = (sub_liste.count(elt)!=0)*1 # test if the current elt is in the  subliste
        results.append(y)
        
    return results

def create_genre_df(data):
    unique_genre = search_genres(data)
    data_genre = data.genres.apply(lambda x:dummy_genre(x, unique_genre))
    data_genre = data_genre.apply(pd.Series)
    data_genre = data_genre.rename(columns=dict(zip(range(len(unique_genre)),unique_genre))) 
    data_genre.index = data.id_app  

    return data_genre

def search_genres_top(data_genre, nb):
    cum_genres = data_genre.sum(axis = 0).sort_values(ascending=False)
    top_genre = list(cum_genres[:nb].index)
    return top_genre


def process_database(data):

    data['contentAdvisoryRating'] = data['contentAdvisoryRating'].apply(lambda x : x.replace('+', '_plus'))
    data['contentAdvisoryRating'] = data['contentAdvisoryRating'].apply(contentAdvisoryRating_recode)

    data.price = pd.to_numeric(data.price, errors='coerce')

    date_end ="2019-12-10" # The date from which the age is computed

    data['age_app'] = data.releaseDate.apply(lambda x : date_diff(x, date_end))
    data['releaseMonth'] = data.releaseDate.apply(get_month)
    data['last_update'] = data.currentVersionReleaseDate.apply(lambda x : date_diff(x, date_end))
    data['nb_language'] = data.languageCodesISO2A.apply(len)

    data_lang = create_lang_df(data)
    top_lang = search_languages_top(data_lang, nb=10) # num top languages = 10
    data = data.merge(data_lang[top_lang], how='inner', left_on='id_app', right_index=True) # Will need to return data[top_lang]

    data['nb_genres'] = data.genres.apply(len)

    data_genre = create_genre_df(data)
    top_genre = search_genres_top(data_genre, nb=20) # num top genres = 20
    data = data.merge(data_genre[top_genre], how='inner', left_on='id_app', right_index=True) # Will need to return data[top_genre]

    sizes = pd.to_numeric(data.fileSizeBytes, errors='coerce')
    sizes = sizes.fillna(0)
    data['fileSizeMB']=sizes.apply(lambda x : np.log(round(x/10**6,2)+1))

    data_exp_dev = data[['sellerName','id_app']].groupby(by='sellerName', as_index=False).count()
    data_exp_dev.rename(columns={'id_app':'Exp_dev'}, inplace = True)
    data = data.merge(data_exp_dev, how='left', on='sellerName') # Will need to return data['Exp_dev']

    return np.c_[np.log( 1 + data['age_app'].values),
                 pd.get_dummies(data[['releaseMonth']]).values,
                 np.log(1 + data['last_update'].values),
                 np.log(1 + data['nb_language'].values),
                 data[top_lang].values,
                 data['nb_genres'].values,
                 data[top_genre].values,
                 np.log( 1 + data['fileSizeMB'].values),
                 np.log( 1 + data['Exp_dev'].values)]




def get_estimator():

    all_col = ['id_app', 'releaseDate', 'artworkUrl512', 'languageCodesISO2A',
       'fileSizeBytes', 'minimumOsVersion', 'trackName',
       'contentAdvisoryRating', 'genres', 'sellerName', 'formattedPrice',
       'price', 'currency', 'version', 'description',
       'currentVersionReleaseDate', 'age_app']

    drop = ['releaseDate', 'artworkUrl512', 'languageCodesISO2A', 'fileSizeBytes', 
        'trackName', 'genres', 'sellerName', 'formattedPrice', 'currency', 
        'description', 'currentVersionReleaseDate']

    database_transformer = FunctionTransformer(
        process_database, validate=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('new_vars_and_process', database_transformer, all_col),
            ('drop cols', 'drop', drop),
        ], remainder='passthrough', n_jobs=-1)
    
    regressor = RandomForestRegressor(
            n_estimators=50, max_depth=20, max_features=.8, n_jobs=-1
        )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', regressor)
    ])

    return pipeline
