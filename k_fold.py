import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.tree import DecisionTreeClassifier     # tree


def make_model(model, parameter):
    if model == 'knn':
        return KNeighborsClassifier(n_neighbors=parameter)
    if model == 'tree':
        return DecisionTreeClassifier(max_depth=parameter)
    else:
        return
        

def k_fold_cv(k, parameter, model, x_learn, y_learn):
    """
    Returns the cross validation error for a given number 
    of subsets k, a given parameter for the considered model
    and a given learning dataset.

    """
    errors = np.zeros(k)
    x_subsets = np.array_split(x_learn, k)
    y_subsets = np.array_split(y_learn, k)
    for x, y, i in zip(x_subsets, y_subsets, range(k)):
        x_to_fit = x_subsets.copy()
        del x_to_fit[i]
        y_to_fit = y_subsets.copy()
        del y_to_fit[i]
        
        x_to_fit = np.concatenate(x_to_fit, axis=0)
        y_to_fit = np.concatenate(y_to_fit, axis=0)
        
        if model == 'knn':
            if parameter >= len(y_to_fit):
                parameter = len(y_to_fit) - 1
        
        model = make_model(model, parameter)
        model.fit(np.concatenate(x_to_fit, axis=0),
                 np.concatenate(y_to_fit, axis=0))
        errors[i] = 1 - model.score(x, y)
    return np.mean(errors), np.var(errors)


