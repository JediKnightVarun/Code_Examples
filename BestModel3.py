# This code creates models for detecting the different types of fatigue.
# It performs a Meta-Learning technique called Random Search to find the best configuration for the model.
# The final model each function returns is the best model over the randomized search ranges.

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import scipy.stats as stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import random
import numpy as np

# K-Nearest Neighbour (KNN) Classifier
def Knn_Best(X,y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Create the range of values over which the random search will be performed
    space = dict()
    space['weights'] = ['uniform', 'distance']
    space['algorithm'] = ['ball_tree', 'kd_tree', 'brute']
    space['n_neighbors'] = random.sample(range(1,20),15)
    space['p'] = random.sample(range(1,20),15)

    model = KNeighborsClassifier(n_jobs=-1)
    search = RandomizedSearchCV(model, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print("")
    print("KNN")
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    # Create the range of values over which the random search will be performed
    # For Bagging method
    space2 = dict()
    space2['n_estimators'] = random.sample(range(50,100),10)
    space2['oob_score'] = [True,False]
    space2['warm_start'] = [True,False]

    model2 = BaggingClassifier(base_estimator= KNeighborsClassifier(n_jobs=-1), n_jobs=-1)
    search2 = RandomizedSearchCV(model2, space2, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,
                                verbose=10)

    result2 = search2.fit(X,y)

    return result, result2

# Random Forrest
def RF_Best(X,y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Create the range of values over which the random search will be performed
    space = dict()
    space['criterion'] = ['gini', 'entropy']
    space['class_weight'] = ['balanced', 'balanced_subsample']
    space['oob_score'] = [True,False]
    space['max_features'] = ['sqrt', 'log2']
    space['n_estimators'] = random.sample(range(5,2000),10)
    space['max_depth'] = random.sample(range(1,100),10)
    space['min_samples_split'] = random.sample(range(2, 10), 5)

    model = RandomForestClassifier( n_jobs= -1)
    search = RandomizedSearchCV(model, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print("")
    print("RF")
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    # Create the range of values over which the random search will be performed
    # For Bagging Method
    space2 = dict()
    space2['n_estimators'] = random.sample(range(50, 100), 10)
    space2['oob_score'] = [True, False]
    space2['warm_start'] = [True, False]

    model2 = BaggingClassifier(base_estimator=RandomForestClassifier( n_jobs= -1), n_jobs=-1)
    search2 = RandomizedSearchCV(model2, space2, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,
                                 verbose=10)

    result2 = search2.fit(X, y)

    return result, result2

# Support Vector Machine
def SVM_Best(X,y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Create the range of values over which the random search will be performed
    space = dict()
    space['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    space['gamma'] = ['scale', 'auto']
    space['degree'] = random.sample(range(3,6),2)
    space['C'] = np.random.uniform(0.01,50.0,20)
    space['probability'] = [True, False]

    model = svm.SVC()
    search = RandomizedSearchCV(model, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print("")
    print("SVM")
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    # Create the range of values over which the random search will be performed
    # For Bagging Method
    space2 = dict()
    space2['n_estimators'] = random.sample(range(50, 100), 10)
    space2['oob_score'] = [True, False]
    space2['warm_start'] = [True, False]

    model2 = BaggingClassifier(base_estimator=svm.SVC(), n_jobs=-1)
    search2 = RandomizedSearchCV(model2, space2, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,
                                 verbose=10)

    result2 = search2.fit(X, y)

    return result, result2

# Naive Bayes
def NB_Best(X,y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Create the range of values over which the random search will be performed
    space = dict()
    space['var_smoothing'] = np.random.uniform(0.0001,1,20)

    model = GaussianNB()
    search = RandomizedSearchCV(model, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print("")
    print("NB")
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    # Create the range of values over which the random search will be performed
    # For Bagging Method
    space2 = dict()
    space2['n_estimators'] = random.sample(range(50, 100), 10)
    space2['oob_score'] = [True, False]
    space2['warm_start'] = [True, False]

    model2 = BaggingClassifier(base_estimator=GaussianNB(), n_jobs=-1)
    search2 = RandomizedSearchCV(model2, space2, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,
                                 verbose=10)

    result2 = search2.fit(X, y)

    return result, result2


# Neural Network
def NN_Best(X,y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Create the range of values over which the random search will be performed
    space = dict()
    space['activation'] = ['identity', 'logistic', 'tanh', 'relu']
    space['solver'] = ['lbfgs', 'sgd', 'adam']#'lbfgs',
    space['learning_rate'] = ['constant', 'invscaling', 'adaptive']
    space['warm_start'] = [True,False]
    space['nesterovs_momentum'] = [True,False]
    space['early_stopping'] = [True,False]
    space['shuffle'] = [True,False]
    space['alpha'] = np.random.uniform(0.0001,0.1,10)
    space['momentum'] = np.random.uniform(0, 1, 10)
    space['validation_fraction'] = np.random.uniform(0, 1, 10)
    space['beta_1'] = np.random.uniform(0, 1, 10)
    space['beta_2'] = np.random.uniform(0, 1, 10)
    space['epsilon'] = np.random.uniform(0, 1, 10)
    space['max_iter'] = random.sample(range(200, 1000), 10)

    space['hidden_layer_sizes'] = [(random.randint(2, 8)) for k in range(8)] + [
        (random.randint(2, 8), random.randint(2, 8)) for k in range(16)]

    model = MLPClassifier()
    search = RandomizedSearchCV(model, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print("")
    print("NN")
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    # Create the range of values over which the random search will be performed
    # For Bagging Method
    space2 = dict()
    space2['n_estimators'] = random.sample(range(50, 100), 10)
    space2['oob_score'] = [True, False]
    space2['warm_start'] = [True, False]

    model2 = BaggingClassifier(base_estimator=MLPClassifier(), n_jobs=-1)
    search2 = RandomizedSearchCV(model2, space2, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,
                                 verbose=10)

    result2 = search2.fit(X, y)

    return result, result2

# Gradiant Boosting
def GB_Best(X,y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Create the range of values over which the random search will be performed
    space = dict()
    space['loss'] = ['deviance', 'exponential']
    space['n_estimators'] = random.sample(range(50,500),50)
    space['criterion'] = ['friedman_mse', 'mse', 'mae']
    space['max_depth'] = random.sample(range(3,100),50)
    space['warm_start'] = [True, False]
    space['n_iter_no_change'] = random.sample(range(10,50),10)

    model = GradientBoostingClassifier()

    search = RandomizedSearchCV(model, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,
                                verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print("")
    print("Gradiant")
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    return result

# Histogram Boosting
def HB_Best(X,y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Create the range of values over which the random search will be performed
    space = dict()
    space['min_samples_leaf'] = random.sample(range(5,20),10)
    space['warm_start'] = [True, False]
    space['n_iter_no_change'] = random.sample(range(10, 50), 10)

    model = HistGradientBoostingClassifier()

    search = RandomizedSearchCV(model, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,
                                verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print("")
    print("Gradiant")
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    return result