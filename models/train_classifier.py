"""
Trains a random forest classifier to predict Figure Eight 
Multilingual Disaster Response Messages data categories

https://www.figure-eight.com/dataset/combined-disaster-response-data/

References:
----------
https://stackoverflow.com/questions/34293875/
    how-to-remove-punctuation-marks-from-a-string-in-
    python-3-x-using-translate/34294022

https://stackoverflow.com/questions/15586721/
    wordnet-lemmatization-and-pos-tagging-in-python

https://stackoverflow.com/questions/40144473/
    do-we-need-to-use-stopwords-filtering-before-pos-tagging

https://stackoverflow.com/questions/27673527/
    how-should-i-vectorize-the-following-list-of-lists-with-scikit-learn

https://stackoverflow.com/questions/34293875/
    how-to-remove-punctuation-marks-from-a-string-in-
    python-3-x-using-translate/34294022

https://stackoverflow.com/questions/43162506/
    undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-
    0-0-in-labels-wi

https://stackoverflow.com/questions/50705190/
    gridsearchcv-cant-pickle-function-error-when-
    trying-to-pass-lambda-in-paramete

https://github.com/scikit-learn/scikit-learn/issues/10054

https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-
    python-using-scikit-learn-28d2aa77dd74    
"""
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
import json
import nltk        
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from TextPreprocessor import TextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

def init_parser():
    """
    Initializes a command line arguments parser

    INPUT:
        None

    OUTPUT:
        parser: ArgumentParser class object
    """
    parser = ArgumentParser(description = "Train classifier script")

    parser.add_argument('database_path',
                        type=str,
                        help='Full path to the Messages SQL database')

    parser.add_argument('model_path',
                        type=str,
                        help='Full path to the model *.pkl file')

    parser.add_argument('--random_state',
                        type=int,
                        default=1011768029,
                        help="Random seed")

    parser.add_argument('--train_size',
                        type=float,
                        default=0.7,
                        help="Fraction of data used for training")

    parser.add_argument('--test_size',
                        type=float,
                        default=0.3,
                        help="Fraction of data used for test")

    parser.add_argument('--gridsearch_parameters_path',
                        type=str,
                        help='Full path to a GridSearch parameters ' +
                             ' *.json file')

    parser.add_argument('--verbose',
                        type=int,
                        default=3,
                        help="sklearn GridSearchCV verbosity level")

    parser.add_argument('--cv',
                        type=int,
                        default=5,
                        help="Number of cross-validation folds")

    parser.add_argument('--n_jobs',
                        type=int,
                        default=3,
                        help="Number of GridSearchCV parallel tasks")

    return parser                   

def train_classifer():
    """                   
    Trains a random forest classifier to predict Figure Eight 
    Multilingual Disaster Response Messages data categories

    INPUT:
        None - Parameters are specified via the command line

    OUTPUT
        None - Trained classifier is written to a *.pkl file
    """
    warnings.filterwarnings("ignore")

    parser = init_parser()

    arguments = parser.parse_args()

    download_nltk_data()

    X, Y = load_cleandata_fromdb(arguments)

    pipeline = init_pipeline(arguments)

    X_train, X_test, Y_train, Y_test =\
        train_test_split(X,
                         Y,
                         random_state=arguments.random_state,
                         train_size=arguments.train_size,
                         test_size=arguments.test_size)

    pipeline.fit(X_train, Y_train)

    Y_pred = pipeline.predict(X_test)

    scoring_df = evaluate_model(Y_test, 
                                Y_pred)

    print("---------------------------------------------------")
    print("Initial model")
    print("---------------------------------------------------")
    print(scoring_df)

    parameters = init_gridsearch_parameters(arguments)

    cv = GridSearchCV(pipeline,
                      parameters,
                      verbose=arguments.verbose,
                      n_jobs=arguments.n_jobs,
                      cv=arguments.cv)

    cv.fit(X_train, Y_train)

    Y_refined_pred = cv.predict(X_test)

    refined_scoring_df = evaluate_model(Y_test, 
                                        Y_refined_pred)

    print("---------------------------------------------------")
    print("Refined model")
    print("---------------------------------------------------")
    print(refined_scoring_df)

    # https://machinelearningmastery.com/
    #   save-load-machine-learning-models-python-scikit-learn/
    with open(arguments.model_path, 'wb') as fp:
        pickle.dump([cv, Y_train.columns], fp)

def download_nltk_data():
    """Downloads Natural Language Toolkit data

    INPUT:
        None

    OUTPUT:
        None
    """
    nltk_dependencies = ['averaged_perceptron_tagger',
                         'maxent_ne_chunker',
                         'punkt',
                         'stopwords',
                         'words',
                         'wordnet',
                         'vader_lexicon']
                        
    for elem in nltk_dependencies:
        nltk.download(elem)

def load_cleandata_fromdb(arguments):
    """Loads clean Figure Eight Multilingual Disaster Response Messages 
    data from an SQL database
    
    INPUT:
        arguments: Namepsace object handle
    
    OUTPUT:
        X: Figure Eight Multilingual Disaster Response Messages

        Y: Figure Eight Multilingual Disaster Response messages categories"""
    engine = create_engine('sqlite:///' + arguments.database_path)
    df = pd.read_sql('SELECT * FROM Messages', engine)
    df.head()

    return df['message'].values, df.iloc[:,4:]

def init_pipeline(arguments):
    """Initializes a machine learning pipeline

    INPUT:
        arguments: Namespace class object handle

    OUTPUT:
        pipeline: Pipeline class object handle"""
    return Pipeline([('preprocess', TextPreprocessor()),
                     ('vectorizer',
                     TfidfVectorizer(ngram_range=(1, 2))),
                     ('clf',
                      RandomForestClassifier(random_state=
                                             arguments.random_state))])

def evaluate_model(Y_test,
                   Y_pred):
    """"Evaluates a model that predicts Figure Eight Multilingual Disaster
    Response Messages data categories

    INPUT:
        Y_test: Expected message categories

        Y_pred: Predicted message categories

    OUTPUT:
        scoring_df: Pandas DataFrame that stores the precision, recall,
                    and f1-score for each predicted message category
    """
    scoring = []

    for idx in range(Y_test.shape[1]):

        report = classification_report(Y_test.iloc[:,idx],
                                       Y_pred[:,idx],
                                       output_dict=True)

        weighted_avg = report['weighted avg']

        scoring.append({'column': Y_test.columns[idx],
                        'precision': weighted_avg['precision'],
                        'recall': weighted_avg['recall'],
                        'f1score': weighted_avg['f1-score']})

    scoring_df = pd.DataFrame(scoring)
    scoring_df = scoring_df.sort_values('f1score', ascending=False)

    return scoring_df.reset_index(drop=True)

def init_gridsearch_parameters(arguments):
    """
    Initializes a dictionary that stores GridSearch parameters

    INPUT:
        arguments: Namespace class object handle

    OUTPUT:
        parameters: Dictionary that stores GridSearch parameters
    """
    if arguments.gridsearch_parameters_path is None:
        parameters = {"vectorizer__max_features": [500, 1000, 2000],
                      "vectorizer__ngram_range": [(1, 2)],
                      "clf__max_features": ['auto', 'log2'],
                      "clf__n_estimators": [50, 100, 150]}
    else:
        with open(arguments.gridsearch_parameters_path, 'rt') as fp:
            parameters = json.load(fp)

    return parameters

if __name__ == "__main__":
    train_classifer()
