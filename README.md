# Kaggle Titanic: Machine Learning Modeling and Deployment
This repository builds a machine learning model on Titanic dataset, creates a web app with Flask to predict the probability of survival per passenger, and deploys the app to Heroku.

## Quick Look
- This video will help you get a feel for the app.  
[<img src="https://img.youtube.com/vi/WlgMhIuC9pY/0.jpg" width="50%">](https://youtu.be/WlgMhIuC9pY). 

- Also you could try the app yourself by clicking the button below.
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://titanic-pred.herokuapp.com/index)  

## Keywords
EDA, custom estimator, tree classifiers, boosting, model persistance, html, flask, Heroku

## Prerequisites
- Datasets  
[Here](https://www.kaggle.com/c/titanic/overview) is the overview of the project.  
- Dependencies  
xgboost==0.90  
Flask==1.1.1  
dill==0.3.0  
numpy==1.16.4  
requests==2.22.0  
pandas==0.25.1  
scikit_learn==0.21.3  
gunicorn==19.9.0  

## Accomplishment  
- EDA, feature engineering and modeling (Titanic.ipynb)  
- Model deployment (#clone the repository and run locally)  
  1. `export FLASK_APP=app.py`  
  2. `python -m flask run`   
  3. Check the link in the terminal
