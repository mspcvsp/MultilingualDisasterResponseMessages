# Disaster Response Pipeline Project

### Instructions:
1. Download the [Figure Eight Multilingual Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) data and move it to the data directory  
  
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

### Files:

### Libraries Used:
