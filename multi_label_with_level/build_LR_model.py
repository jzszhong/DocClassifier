from build_models import *

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def build_LR_model(X_train, X_test, Y_train, Y_test):
    print('Training Logistic Regression model')
    print('------------------------------')

    params = {
        'C': [10**i for i in range(-5, 5)],
        'tol': [10**i for i in range(-10, 3)]
    }

    model = LogisticRegression()

    # Cross validation and search for the best hyperparameters
    cv = GridSearchCV(param_grid=params, estimator=model, cv=10, n_jobs=-1)
    cv.fit(X_train, Y_train)

    model = cv.best_estimator_
    print(cv.best_params_)

    # Outputs the accuracies
    print("Train accuracy:", model.score(X_train, Y_train))
    print("Test accuracy:", model.score(X_test, Y_test))
    print(classification_report(Y_test, model.predict(X_test)))

    return model, model.score(X_test, Y_test), model.predict(X_test)

if __name__ == '__main__':
    build_models_with_level(build_LR_model, "LR")