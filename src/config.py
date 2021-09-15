import os
import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
# import xgboost as xgb
import flask
from flask import Flask, request, jsonify
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from numpy import genfromtxt
from flask_migrate import Migrate
from sqlalchemy.orm import scoped_session, sessionmaker
import datetime
import logging
from sqlalchemy import desc
from flask import abort
import re
import json


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgres@localhost:5432/issues_api"
#initialize db
db = SQLAlchemy(app)
ENTRY_POINT = '/api'


HOME_DIR = os.path.expanduser("~")
TRAINING_FILE =  os.path.join(HOME_DIR, "data", "finaldf.csv")
DATABASE_FILE =  os.path.join(HOME_DIR, "data", "complete_df.csv")

#initialize log file
logging.basicConfig(filename=os.path.join(HOME_DIR,'log.log'))

