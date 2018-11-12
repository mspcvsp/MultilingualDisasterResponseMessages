# Disaster Response Pipeline Project

### Project Motivation
Construct a classifier that predicts [Figure Eight multilingual disaster response message](https://www.figure-eight.com/dataset/combined-disaster-response-data/) categories.  
  
### Instructions:  
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:  
- app  
  - run.py: Web application backend implemented using [Flask](http://flask.pocoo.org/)  
  - templates  
    - master.html - Web application front end  
    - go.html - Defines web page that is rendered when a user requests the prediction of message categories  
- data
  - disaster_categories.csv - Comma Separated Value (CSV) file that stores disaster response message categories  
  - disaster_messages.csv - Comma Separated Value (CSV) file that stores disaster response messages  
  - process_data.py - Cleans the [Figure Eight multilingual disaster response messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) dataset and stores it in an SQLLite database  
- models  
  - TextPreprocessor.py - Custom [tokenzier](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-tokenizers.html) that is placed in a separate *.py file in order to run [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) in parallel. 
  - train_classifier.py - Trains a [random forest classifier](https://en.wikipedia.org/wiki/Random_forest) to predict [Figure Eight multilingual disaster response message](https://www.figure-eight.com/dataset/combined-disaster-response-data/) categories.  
  
### Libraries Used:  
- [argparse](https://docs.python.org/3/library/argparse.html)  
- [flask](http://flask.pocoo.org)  
- [json](https://docs.python.org/3/library/json.html)  
- [nltk](https://www.nltk.org)  
- [numpy](http://www.numpy.org)  
- [pandas](https://pandas.pydata.org)  
- [plotly](https://plot.ly/python/)  
- [re](https://docs.python.org/3/library/re.html)  
- [sklearn](https://scikit-learn.org/stable/)  
- [sqlalchemy](https://www.sqlalchemy.org)  
- [textblob](https://textblob.readthedocs.io/en/dev/)  
- [warnings](https://docs.python.org/3.7/library/warnings.html)  
