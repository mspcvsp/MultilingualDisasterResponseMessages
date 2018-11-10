"""
Cleans the Figure Eight Multilingual Disaster Response Messages data

https://www.figure-eight.com/dataset/combined-disaster-response-data/
"""

# import libraries
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine

def init_parser():
    """
    Initializes a command line arguments parser

    INPUT:
        None

    OUTPUT:
        parser: ArgumentParser class object
    """
    parser = ArgumentParser(description = "Extract Transform Load script")

    parser.add_argument('messages_dataset_path',
                        type=str,
                        help='Full path to the messages dataset *.csv')

    parser.add_argument('categories_dataset_path',
                        type=str,
                        help='Full path to the categories dataset *.csv')

    parser.add_argument('database_path',
                        type=str,
                        help='Full path to the Messages SQL database')

    return parser

def process_data():
    """
    Cleans the Figure Eight Multilingual Disaster Response Messages data

    INPUT:
        None - messages and categories dataset paths are specified via the
               command line

    OUTPUT:
        None - Cleaned dataset is written to an SQLlite database
    """
    parser = init_parser()

    arguments = parser.parse_args()

    # load messages dataset
    messages = pd.read_csv(arguments.messages_dataset_path)

    # load categories dataset
    categories = pd.read_csv(arguments.categories_dataset_path)

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

    # Filter invalid category labels
    select_data =\
        ((df.iloc[:,4:].values > 1).sum(axis=1) == 0).astype('bool')

    percent_filtered =\
        100 * (1 - (np.count_nonzero(select_data) / len(select_data)))

    print("Percentage of invalid category label(s): %.2f%%" %\
        (percent_filtered))

    df = df[select_data]

    # Drop the Messages table if it exists
    #
    # https://docs.sqlalchemy.org/en/latest/core/engines.html
    #
    # https://stackoverflow.com/questions/33229140/
    #   how-do-i-drop-a-table-in-sqlalchemy-when-i-dont-have-a-table-object
    engine = create_engine('sqlite:///' + arguments.database_path)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    command = "DROP TABLE IF EXISTS {};".format("Messages")
    cursor.execute(command)
    connection.commit()
    cursor.close()

    # Save the clean dataset into an sqlite database.
    df.to_sql('Messages', engine, index=False)    

def clean_category_names(categories):
    """The format of each categories data frame entry is <category>-<0|1>
    where <category> is a string followed by a 0 or 1 integer value.
    
    This function creates a set of clean category names by applying
    two transformations:
    
    1.) Splitting a category string using the '-' character
    2.) Removing underscores
    
    INPUT:
        categories: Pandas DataFrame that stores disaster message categories

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
