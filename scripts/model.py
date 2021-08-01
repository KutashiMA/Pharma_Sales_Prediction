
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

import logging
import dvc.api 

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

path ='data/cleaned_data.csv'
repo="C:/Users/Alt/workspace/Python/10academy/Week-3/Pharma_Sales_Prediction"
version="v1"


data_url= dvc.api.get_url(
        path=path,
        repo=repo,
        rev=version)
#mlflow.set_experiment('dvc')


def eval_metrics(actual, pred):
    
    rmse=np.sqrt(mean_squared_error(actual, pred))
    
    return rmse

def model_select(model):
    if model == 'RandomForestRegressor':
        from sklearn.ensemble import RandomForestRegressor
        mod = RandomForestRegressor()

    elif model == 'DecisionTreeRegressor':
        from sklearn.tree import DecisionTreeRegressor
        mod = DecisionTreeRegressor()
        
    elif model == 'GradientBoostingRegressor':
        from sklearn.ensemble import GradientBoostingRegressor
        mod = GradientBoostingRegressor()
        
    return mod, model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Reading the csv file from the URL
    csv_url = (
        data_url
    )
    try:
        data = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to load dataset", e
        )
    
    # necessary data 
    train_X=data[['Store', 'DayOfWeek', 'Customers','Promo', 'year', 'month']][:100000]
    train_Y=data['Sales'][:100000]
    lb = LabelEncoder()
    train_X['month'] = lb.fit_transform(train_X['month'])
    # Split the data into training test sets..
    train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y,test_size=0.2,random_state=0)



     # log artifacts columns 
    cols_x=pd.DataFrame(list(data[['Store', 'DayOfWeek', 'Customers','Promo', 'year', 'month']].columns))
    cols_x.to_csv('../data/features.csv', header=False, index=False)
    #mlflow.log_artifact('feature.csv')
    
    cols_y=pd.DataFrame(list(data[['Sales']].columns))
    cols_y.to_csv('../data/target.csv', header=False, index=False)
    #mlflow.log_artifact('target.csv')
    
    
    #learning parameters 
#     alpha = 0.8
#     l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    mlflow.end_run()
    with mlflow.start_run():
        model = model_select('GradientBoostingRegressor')
        name = model[1]
        model=model[0]
        model.fit(train_x, train_y)

        predicted_answer = model.predict(test_x)

        rmse= eval_metrics(test_y, predicted_answer)

        print(f"model is :{name}")
        print(f"RMSE: {rmse}")

        mlflow.log_param("model", name)
        mlflow.log_metric("RMSE", rmse)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(predicted_answer[0], "model", registered_model_name=name+" pharma_sales")
        else:
            mlflow.sklearn.log_model(predicted_answer[0], "model")
