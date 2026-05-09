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
    total = len(hyperparameter_grid)

    for i, hyperparameters in enumerate(hyperparameter_grid, 1):
        print(f"      [HP {i}/{total}] {hyperparameters} ... ", end="", flush=True)
        model = model_class(**hyperparameters)
        scores = k_fold_cross_validation(model, X, y, k_inner)
        average_score = sum(scores) / len(scores)
        results.append((hyperparameters, average_score))
        print(f"avg acc: {average_score * 100:.2f}%")

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

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(X), 1):
        print(f"  [Outer fold {fold_idx}/{k_outer}]", flush=True)
        X_train_outer, X_test_outer = X[outer_train_idx], X[outer_test_idx]
        y_train_outer, y_test_outer = y[outer_train_idx], y[outer_test_idx]

        # run inner CV only on the outer training set to get hyperparameter evaluations and best combo
        inner_results, best_hyperparameters, _ = inner_k_fold_cross_validation(
            model_class, X_train_outer, y_train_outer, hyperparameter_grid, k_inner
        )

        print(f"    Best HPs: {best_hyperparameters}")

        # train the model with the selected hyperparameters on the full outer training set
        best_model = model_class(**best_hyperparameters)
        best_model.fit(X_train_outer, y_train_outer)

        # evaluate on the outer test set
        predictions = best_model.predict(X_test_outer)
        test_score = accuracy_score(y_test_outer, predictions)
        print(f"    Outer fold score: {test_score * 100:.2f}%", flush=True)

        outer_test_scores.append(test_score)
        selected_hyperparameters.append(best_hyperparameters)

    return outer_test_scores, selected_hyperparameters



def k_fold_n_models(models, X, y, k_outer=5, k_inner=5):
    """
    Perform nested outer evaluation for multiple models.

    The outer loop iterates over folds; within each fold all candidate models
    are evaluated with inner CV for hyperparameter tuning, then retrained on
    the full outer training set and scored on the outer test fold.
    This matches the nested CV procedure described in the assignment (Part D).

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
    outer_kf = KFold(n_splits=k_outer, shuffle=True, random_state=42)

    # initialise per-model storage
    results = {
        name: {'outer_test_scores': [], 'selected_hyperparameters': []}
        for name, _, _ in models
    }

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(X), 1):
        print(f"\n{'=' * 50}", flush=True)
        print(f"[Outer fold {fold_idx}/{k_outer}]", flush=True)
        print(f"{'=' * 50}", flush=True)

        X_train_outer, X_test_outer = X[outer_train_idx], X[outer_test_idx]
        y_train_outer, y_test_outer = y[outer_train_idx], y[outer_test_idx]

        # evaluate every candidate model within this fold
        for model_name, model_class, hyperparameter_grid in models:
            print(f"\n  Model: {model_name}", flush=True)

            # inner CV on the outer training set to select the best hyperparameters
            _, best_hyperparameters, _ = inner_k_fold_cross_validation(
                model_class, X_train_outer, y_train_outer, hyperparameter_grid, k_inner
            )

            print(f"    Best HPs: {best_hyperparameters}")

            # retrain on the full outer training set with the selected hyperparameters
            best_model = model_class(**best_hyperparameters)
            best_model.fit(X_train_outer, y_train_outer)

            # evaluate on the outer test fold
            predictions = best_model.predict(X_test_outer)
            test_score = accuracy_score(y_test_outer, predictions)
            print(f"    Outer fold score: {test_score * 100:.2f}%", flush=True)

            # store this fold's score and chosen hyperparameters for this model
            results[model_name]['outer_test_scores'].append(test_score)
            results[model_name]['selected_hyperparameters'].append(best_hyperparameters)

    # compute summary statistics across all outer folds
    for model_name in results:
        scores = results[model_name]['outer_test_scores']
        results[model_name]['avg_score'] = sum(scores) / len(scores) if scores else None
        try:
            results[model_name]['variance'] = statistics.pvariance(scores)
        except Exception:
            results[model_name]['variance'] = None

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
