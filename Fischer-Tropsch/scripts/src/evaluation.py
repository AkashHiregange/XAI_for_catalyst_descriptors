import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


def evaluate_models(features,properties,estimator,best_parameters,config,model_name):
    """This function fits the tuned desired model and returns the train/test metrics per target."""

    config.create_directories()

    results = []

    for target in properties.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            features,properties[target],test_size=config.test_size,random_state=config.random_state)

        model = clone(estimator).set_params(**best_parameters[target])
        model.fit(X_train, y_train)

        for split, X_part, y_part in (("train", X_train, y_train),("test", X_test, y_test)):
            predictions = model.predict(X_part)

            results.append({"model": model_name,"target": target,"split": split,"r2": r2_score(y_part, predictions),
                            "mae": mean_absolute_error(y_part, predictions),
                            "rmse": mean_squared_error(y_part, predictions) ** 0.5})

        model_path = (f"{config.output_dir}/{model_name}_{target}_model.pkl")

        with model_path.open("wb") as file:
            pickle.dump(model, file)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{config.output_dir}/{model_name}_metrics.csv",index=False)

    return results_df