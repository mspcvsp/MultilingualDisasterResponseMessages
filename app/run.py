"""
Backend for a web application that:

1.) Predicts disaster response message categories

2.) Generates two plots that describe disaster response message
category prediction training data

Data set URL:
------------
https://www.figure-eight.com/dataset/combined-disaster-response-data/
"""

import sys
import json
import plotly
from plotly import graph_objs
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from textblob import TextBlob

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
sys.path.append("../models")
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Renders the web application index page"""

    # extract data needed for visuals
    training_data = init_training_data(df,
                                       model)

    category_percentage_df = init_categorypercentage_df(training_data)

    sentiment_df = evaluate_category_sentiment('water',
                                               training_data)

    # create visuals
    graphs = [init_categorypercentage_plot(category_percentage_df),
              init_sentiment_histogram(sentiment_df)]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """Runs the web application"""
    app.run(host='0.0.0.0', port=3001, debug=True)

def init_training_data(df,
                       model):
    """Initializes a data frame that stores the training data
    
    INPUT:
        df: Pandas DataFrame that stores Figure Eight Multilingual Disaster
            Response Messages

        model: scikit-learn classifier object handle

    OUTPUT:
        training_data_df: Pandas DataFrame that stores training data"""
    params = model.get_params()
    params = params['estimator__clf'].get_params()

    X_train, _, Y_train, _ =\
        train_test_split(df['message'].values,
                         df.iloc[:,4:],
                         random_state=params['random_state'],
                         train_size=0.7,
                         test_size=0.3)

    train_messages = pd.DataFrame(X_train, columns=['message'])

    Y_train = Y_train.reset_index(drop=True)

    return pd.concat([train_messages, Y_train], axis=1)

def init_categorypercentage_df(training_data_df):
    """Initializes a Pandas DataFrame that stores Figure Eight 
    multilingual disaster response message category percentages
    
    INPUT:
        training_data_df: Pandas DataFrame that stores training data

    OUTPUT:
        category_percentage_df: Pandas DataFrame that stores Figure Eight 
                                multilingual disaster response message 
                                category percentages
    """
    normalization = 100 / training_data_df.shape[0]

    category_percentage = []

    for key in training_data_df.columns[1:]:
        category_percentage.append((key,
                                    normalization *
                                    training_data_df[key].sum()))
    
    category_percentage_df =\
        pd.DataFrame(category_percentage,
                     columns=['category', 'percentage'])

    category_percentage_df =\
        category_percentage_df.sort_values('percentage',
                                           ascending=False)

    return category_percentage_df.reset_index(drop=True)

def evaluate_category_sentiment(category,
                                training_data_df):
    """Evaluates the sentiment of a category
    
    INPUT:
        category: String that refers to a specific message category
        
        training_data_df: Pandas DataFrame that stores training data

    OUTPUT:
        sentiment_df: Pandas DataFrame that stores the estimated
                      sentiment for a specific message category"""
    categorydata_df =\
        training_data_df[training_data_df[category] == 1].copy()

    categorydata_df = categorydata_df.reset_index(drop=True)
    
    sentiment = []
    for idx in range(categorydata_df.shape[0]):
        tb = TextBlob(categorydata_df.loc[idx, 'message'])
        sentiment.append((tb.polarity, tb.subjectivity))

    return pd.DataFrame(sentiment, columns=['polarity','subjectivity'])

def init_categorypercentage_plot(category_percentage_df):
    """Initializes a training data message category percentages
    horizontal bar chart
    
    INPUT:
        category_percentage_df: Pandas DataFrame that stores Figure Eight 
                                multilingual disaster response message 
                                category percentages

    OUTPUT:
        figureobj: Figure class object handle"""
    data = [graph_objs.Bar(x=category_percentage_df['percentage'].values,
                           y=category_percentage_df['category'].values,
                           orientation = 'h')]

    layout =\
        graph_objs.Layout(title='Training Data<br>Category Percentage',
                          font=dict(family='Courier New, monospace',
                                    size=16,
                                    color='#7f7f7f'),
                          xaxis=dict(title='Percentage'),
                          yaxis=dict(title='Category'),
                          width=500,
                          height=500,
                          margin=graph_objs.layout.Margin(l=250,
                                                          r=50,
                                                          b=100,
                                                          t=100,
                                                          pad=4))

    return graph_objs.Figure(data=data, layout=layout)

def init_sentiment_histogram(sentiment_df):
    """
    Generates a water message sentiment scatter plot

    INPUT:
        sentiment_df: Pandas DataFrame that stores the estimated
                      sentiment for a specific message category

    OUTPUT:
        figureobj: Figure class object handle 
    """
    data = [graph_objs.Scatter(x=sentiment_df['subjectivity'],
                               y=sentiment_df['polarity'],
                               mode='markers')]

    layout =\
        graph_objs.Layout(title='Water Message<br>Sentiment',
                          font=dict(family='Courier New, monospace',
                          size=16,
                          color='#7f7f7f'),
                          xaxis=dict(title='Subjectivity'),
                          yaxis=dict(title='Polarity'),
                          width=500,
                          height=500,
                          margin=graph_objs.layout.Margin(l=70,
                                                          r=50,
                                                          b=100,
                                                          t=100,
                                                          pad=4),
                          barmode='overlay')

    return graph_objs.Figure(data=data, layout=layout)

if __name__ == '__main__':
    main()
