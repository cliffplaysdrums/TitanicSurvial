from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier


def search_forest_params(features, labels):
    # Per scikit-learn:
    # When using ensemble methods base upon bagging, i.e. generating new training sets using sampling with replacement,
    # part of the training set remains unused. For each classifier in the ensemble, a different part of the training
    # set is left out.
    #
    # This left out portion can be used to estimate the generalization error without having to rely on a separate
    # validation set. This estimate comes “for free” as no additional data is needed and can be used for model
    # selection.
    forest_model = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=1)

    # param_grid = [{'max_depth': list(range(1, 5)),
    #                'n_estimators': [2 ** i for i in range(4, 11)],
    #                'criterion': ['gini', 'entropy']}]
    # I used the above first and found all the top results to be at the depth limit with 'entropy', so let's try again
    # with a more focused search on higher depth & a single criterion option
    param_grid = [{'max_depth': list(range(4, 10)),
                   'n_estimators': [2 ** i for i in range(4, 11)],
                   'criterion': ['entropy']}]
    search = GridSearchCV(forest_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=8)
    search.fit(features, labels)
    print(search.cv_results_)
    print(search.best_estimator_)
    # RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=64, random_state=1)
