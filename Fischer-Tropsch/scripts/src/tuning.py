import json
from pathlib import Path

from sklearn.model_selection import GridSearchCV, KFold, train_test_split


def tune_models(features,properties,estimator,param_grid,config,model_name):
    """This function tunes one regressor independently for each target - pretty sure this can be used for a single-target regression as well"""

    config.create_directories()

    X_train, _ , _ , _ = train_test_split(
        features,properties,test_size=config.test_size,random_state=config.random_state)

    cv = KFold(n_splits=config.cv_folds,shuffle=True,random_state=config.random_state)

    best_parameters = {}
    cv_scores = {}

    for target in properties.columns:
        print('--'*15)
        print(f'tuning for target property {target}')
        print('--'*15)
        y = properties[target]
        #consitent y_train selection
        y_train = y.loc[X_train.index]

        search = GridSearchCV(
            estimator=estimator,param_grid=param_grid,scoring="neg_mean_absolute_error",cv=cv,n_jobs=-1)

        search.fit(X_train, y_train)

        best_parameters[target] = search.best_params_
        #cv_scores[target] = search.best_score_

        print(f"{target}: best parameters = {search.best_params_}")

    output_path = f"{config.output_dir}_{model_name}_best_parameters.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(best_parameters, file, indent=4)

    return best_parameters