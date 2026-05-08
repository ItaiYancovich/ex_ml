from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import statistics
from collections import Counter


def k_fold_cross_validation(model, X, y, k=5):
    """
    Perform k-fold cross-validation on the given model and dataset.

    Parameters:
    model: The machine learning model to be evaluated.
    X: The feature dataset (numpy array or pandas DataFrame).
    y: The target labels (numpy array or pandas Series).
    k: The number of folds for cross-validation (default is 5).

    Returns:
    A list of evaluation scores for each fold.
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        scores.append(score)

    return scores


def inner_k_fold_cross_validation(model_class, X, y, hyperparameter_grid, k_inner=5):
    """
    Perform inner k-fold cross-validation for hyperparameter tuning.

    Parameters:
    model_class: The class of the machine learning model to be evaluated.
    X: The feature dataset (numpy array or pandas DataFrame).
    y: The target labels (numpy array or pandas Series).
    hyperparameter_grid: A list of dictionaries containing hyperparameter combinations.
    k_inner: The number of folds for inner cross-validation (default is 5).

    Returns:
    A tuple containing:
      - results: list of tuples (hyperparameters_dict, average_score)
      - best_hyperparameters: the hyperparameter dict with highest average score
      - best_score: the corresponding average score
    """

    best_hyperparameters = None
    best_score = -float('inf')
    results = []

    for hyperparameters in hyperparameter_grid:
        model = model_class(**hyperparameters)
        scores = k_fold_cross_validation(model, X, y, k_inner)
        average_score = sum(scores) / len(scores)
        results.append((hyperparameters, average_score))

        if average_score > best_score:
            best_score = average_score
            best_hyperparameters = hyperparameters

    return results, best_hyperparameters, best_score


def outer_fold_test(model_class, X, y, hyperparameter_grid, k_outer=5, k_inner=5):
    """
    Perform nested (outer) k-fold testing.

    Parameters:
    model_class: The class of the machine learning model to be evaluated.
    X: The feature dataset (numpy array or pandas DataFrame).
    y: The target labels (numpy array or pandas Series).
    hyperparameter_grid: A list of dictionaries containing hyperparameter combinations.
    k_outer: The number of outer folds.
    k_inner: The number of inner folds used inside each outer split.

    Returns:
    A tuple containing:
      - outer_test_scores: list of evaluation scores (accuracy) for each outer fold
      - selected_hyperparameters: list of best hyperparameter dicts chosen for each outer fold
    """
    outer_kf = KFold(n_splits=k_outer, shuffle=True, random_state=42)
    outer_test_scores = []
    selected_hyperparameters = []

    for outer_train_idx, outer_test_idx in outer_kf.split(X):
        X_train_outer, X_test_outer = X[outer_train_idx], X[outer_test_idx]
        y_train_outer, y_test_outer = y[outer_train_idx], y[outer_test_idx]

        # run inner CV only on the outer training set to get hyperparameter evaluations and best combo
        inner_results, best_hyperparameters, _ = inner_k_fold_cross_validation(
            model_class, X_train_outer, y_train_outer, hyperparameter_grid, k_inner
        )

        # train the model with the selected hyperparameters on the full outer training set
        best_model = model_class(**best_hyperparameters)
        best_model.fit(X_train_outer, y_train_outer)

        # evaluate on the outer test set
        predictions = best_model.predict(X_test_outer)
        test_score = accuracy_score(y_test_outer, predictions)

        outer_test_scores.append(test_score)
        selected_hyperparameters.append(best_hyperparameters)

    return outer_test_scores, selected_hyperparameters



def k_fold_n_models(models, X, y, k_outer=5, k_inner=5):
    """
    Perform nested outer evaluation for multiple models.

    Parameters:
    models: A list of tuples (model_name, model_class, hyperparameter_grid).
            - model_name: string identifier
            - model_class: class (not instance) of the model to instantiate
            - hyperparameter_grid: list of dicts with hyperparameter combinations for inner CV
    X: The feature dataset (numpy array or pandas DataFrame).
    y: The target labels (numpy array or pandas Series).
    k_outer: number of outer folds
    k_inner: number of inner folds
    """
    results = {}

    for model_entry in models:
        # expect (model_name, model_class, hyperparameter_grid)
        model_name, model_class, hyperparameter_grid = model_entry

        # run nested outer evaluation for this model using the provided hyperparameter grid
        outer_test_scores, selected_hyperparameters = outer_fold_test(
            model_class, X, y, hyperparameter_grid, k_outer, k_inner
        )

        avg_score = None
        variance = None
        if outer_test_scores:
            avg_score = sum(outer_test_scores) / len(outer_test_scores)
            # population variance; returns 0.0 for single-value lists
            try:
                variance = statistics.pvariance(outer_test_scores)
            except Exception:
                variance = None

        results[model_name] = {
            'avg_score': avg_score,
            'variance': variance,
            'outer_test_scores': outer_test_scores,
            'selected_hyperparameters': selected_hyperparameters
        }

    return results


def find_best_model(models, X, y, k_outer=5, k_inner=5):
    """
    Perform nested outer evaluation for multiple models.

    Parameters:
    models: A list of tuples (model_name, model_class, hyperparameter_grid).
    k_outer: number of outer folds
    k_inner: number of inner folds

    Returns:
    model with best outer average score and its hyperparameters
    """
    results = k_fold_n_models(models, X, y, k_outer, k_inner)

    best_model_name = None
    best_avg_score = -float('inf')
    best_hyperparameters = None

    for model_name, result in results.items():
        avg_score = result['avg_score']
        if avg_score is not None and avg_score > best_avg_score:
            best_avg_score = avg_score
            best_model_name = model_name
            best_hyperparameters = result['selected_hyperparameters']

    return best_model_name, best_avg_score, best_hyperparameters

def main(models, X_train, y_train, X_test=None, k_outer=5, k_inner=5):
    """
    Perform nested outer evaluation for multiple models and return summary stats and predictions.

    Returns a tuple:
      - best_model_name: name of the model with highest average outer score
      - results: dict mapping model_name -> {
            'avg_score': float or None,
            'variance': float or None,
            'selected_hyperparameters': list of hyperparameter dicts chosen per outer fold,
            'outer_test_scores': list of outer fold scores
        }
      - predictions: predictions on X_test produced by the best model trained on full training data
                     (None if X_test is None)
    """
    # obtain per-model nested CV results
    results = k_fold_n_models(models, X_train, y_train, k_outer, k_inner)

    # determine best model by avg_score
    best_model_name = None
    best_avg = -float('inf')
    for name, res in results.items():
        avg = res['avg_score']
        if avg is not None and avg > best_avg:
            best_avg = avg
            best_model_name = name

    # find the class and per-fold hyperparameters for the best model
    best_model_class = None
    best_selected_hyperparams = []
    for model_entry in models:
        if model_entry[0] == best_model_name:
            best_model_class = model_entry[1]
            # if results contain selected_hyperparameters, use those
            best_selected_hyperparams = results.get(best_model_name, {}).get('selected_hyperparameters', []) or []
            break

    # choose final hyperparameters for training on full training data:
    # use the most common hyperparameter dict across outer folds (mode).
    final_hyperparams = {}
    if best_selected_hyperparams:
        # convert dicts to hashable tuples for counting
        tuples = [tuple(sorted(d.items())) for d in best_selected_hyperparams]
        most_common_tuple, _ = Counter(tuples).most_common(1)[0]
        final_hyperparams = dict(most_common_tuple)

    # train best model on full training data and predict on X_test (if provided)
    predictions = None
    if best_model_class is not None:
        try:
            model_instance = best_model_class(**final_hyperparams) if final_hyperparams else best_model_class()
            model_instance.fit(X_train, y_train)
            if X_test is not None:
                predictions = model_instance.predict(X_test)
        except Exception:
            # if instantiation/training fails, keep predictions as None
            predictions = None

    return best_model_name, results, predictions

if __name__ == "__main__":
    # avoid calling main() without arguments
    pass
