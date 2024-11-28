import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.smpickle import load_pickle
from fastapi import FastAPI

# os.chdir('C:/Users/darre/OneDrive/Documents/My Career/Toolkit - Data Science/Projects/ml_deploy_basic/deployment_folder')

# load model
results_linear = load_pickle("model.pickle")

# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'rate-search', 'description': "Rate API Deployment."}

@app.get("/search")
def search(query: float):
    query_holder = pd.DataFrame({'const': [1],
                       'Miles': [query]})
    pred_result = round(results_linear.predict(query_holder)[0],2)
    return {'result_type': 'predicted_rate', 'rate': pred_result}

