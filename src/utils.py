import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np 
import pandas as pd
import dill
import pickle
from exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,params):
    try:
        report = {}

        for model_key in models.keys():
            model = models[model_key]
            if model_key not in params:
                raise ValueError(f"Model key '{model_key}' not found in params dictionary.")
            params_for_model = params[model_key]

            gs = GridSearchCV(model, params_for_model, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_key] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    