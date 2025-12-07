from numpy import ndarray


def compute_acc(y_true: ndarray, y_prediction: ndarray) -> float:

    # Check if sizes match
    nb_samples = y_true.size
    if y_prediction.size != nb_samples:
        raise ValueError("y_true and y_prediction msut have the same size")
    
    nb_right_predictions = (y_true == y_prediction).sum()

    return nb_right_predictions / nb_samples
