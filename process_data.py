"""
Cleans the Figure Eight Multilingual Disaster Response Messages data

https://www.figure-eight.com/dataset/combined-disaster-response-data/
"""

# import libraries
import pandas as pd
import re
from sqlalchemy import create_engine

def process_data():
    # load messages dataset
    messages = pd.read_csv('messages.csv')

    # load categories dataset
    categories = pd.read_csv('categories.csv')

    # merge datasets
    df = messages.merge(categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    categories.columns = clean_category_names(categories)

    # Convert category values to numbers
    for column in categories:
        categories[column] =\
            categories[column].apply(lambda elem: parse_category_value(elem))

    # Replace `categories` column in `df` with new category columns.
    #
    # https://stackoverflow.com/questions/13411544/
    #   delete-column-from-pandas-dataframe-using-del-df-column-name
    df = df.drop('categories', 1)

    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    #
    # https://stackoverflow.com/questions/23667369/
    #   drop-all-duplicate-rows-in-python-pandas
    df = df.drop_duplicates()

    assert df.duplicated().sum() == 0, "Duplicates not removed"

    # Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('Messages', engine, index=False)    

def clean_category_names(categories):
    """The format of each categories data frame entry is <category>-<0|1>
    where <category> is a string followed by a 0 or 1 integer value.
    
    This function creates a set of clean category names by applying
    the following two transformattions:
    
    1.) Splitting a category string using the '-' character
    2.) Removing underscores
    
    INPUT:
        categories: Pandas DataFrame that stores diaster message categories

    OUTPUT:
        category_colnames: Clean category column names"""
    row = categories.iloc[0]

    category_colnames = row.apply(lambda elem: re.split('-', elem)[0])

    return [re.sub('_', '', elem) for elem in category_colnames]

def parse_category_value(category_value_str):
    """The format of each categories data frame entry is <category>-<0|1>
    where <category> is a string followed by a 0 or 1 integer value.

    This function returns an integer value that corresponds to whether
    a message is associated with a category.

    INPUT:
        category_value_str: Categories data frame entry

    OUTPUT:
        category_value: Integer value that corresponds to whether a
                        message is associated with a category"""
    matchobj = re.match('^[a-z_]+-([0-9])$', category_value_str)

    if matchobj is None:
        raise ValueError('Invalid category value: %s' % (category_value_str))

    return int(matchobj.groups(1)[0])

if __name__ == "__main__":
    process_data()
