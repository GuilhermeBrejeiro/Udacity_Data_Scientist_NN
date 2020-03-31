# Disaster Response Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [Instructions](#instructions)

## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk:
* punkt
* wordnet
* stopwords
* averaged_perceptron_tagger


## Project Motivation<a name="motivation"></a>

Due to the high number of open events on various topics, this project consists of applying machine learning and NLP techniques to separate and distribute them in the correct categories, facilitating their dealings. In addition to the applied techniques, pipelines were also used to facilitate and organize the project.


## File Descriptions <a name="files"></a>

There are three main foleders:
1. data
    - disaster_categories.csv: dataset of all categories 
    - disaster_messages.csv: dataset of all messages
    - process_data.py: ETL pipeline to read, clean, and save database
    - DisasterResponse.db: output of the ETL pipeline
2. models
    - train_classifier.py: ML pipeline to train and export the classifier
    - classifier.pkl: output of the machine learning pipeline
3. app
    - run.py: Flask file for web application
    - templates: contains the html file for web applicatin

## Results<a name="results"></a>

1. Pipleline built to read data from two csv files, clean and save them into a SQLite database.
2. ML pipepline to train a classifier to performs mult-classification on the 36 categories of dataset.
3. A Flask app to show data visualization with top 3 most and lowest occurences and classify the message that user enters on the web page.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to Udacity for the starter codes and the data providade for this project. 
## Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory.

    - To clean data and store in database:  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To train classifier and saves it:  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Link Udacity Workspace: https://view6914b2f4-3001.udacity-student-workspaces.com/
