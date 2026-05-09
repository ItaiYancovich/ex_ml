from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import statistics
from collections import Counter


def k_fold_cross_validation(model, X, y, k=5):
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


def k_fold_n_models(models, X, y, k_outer=5, k_inner=5):
    outer_kf = KFold(n_splits=k_outer, shuffle=True, random_state=42)
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

        for model_name, model_class, hyperparameter_grid in models:
            print(f"\n  Model: {model_name}", flush=True)

            _, best_hyperparameters, _ = inner_k_fold_cross_validation(
                model_class, X_train_outer, y_train_outer, hyperparameter_grid, k_inner
            )

            print(f"    Best HPs: {best_hyperparameters}")

            best_model = model_class(**best_hyperparameters)
            best_model.fit(X_train_outer, y_train_outer)

            predictions = best_model.predict(X_test_outer)
            test_score = accuracy_score(y_test_outer, predictions)
            print(f"    Outer fold score: {test_score * 100:.2f}%", flush=True)

            results[model_name]['outer_test_scores'].append(test_score)
            results[model_name]['selected_hyperparameters'].append(best_hyperparameters)

    for model_name in results:
        scores = results[model_name]['outer_test_scores']
        results[model_name]['avg_score'] = sum(scores) / len(scores) if scores else None
        try:
            results[model_name]['variance'] = statistics.pvariance(scores)
        except Exception:
            results[model_name]['variance'] = None

    return results


def main(models, X_train, y_train, X_test=None, k_outer=5, k_inner=5):
    results = k_fold_n_models(models, X_train, y_train, k_outer, k_inner)

    best_model_name = None
    best_avg = -float('inf')
    for name, res in results.items():
        avg = res['avg_score']
        if avg is not None and avg > best_avg:
            best_avg = avg
            best_model_name = name

    best_model_class = None
    best_selected_hyperparams = []
    for model_entry in models:
        if model_entry[0] == best_model_name:
            best_model_class = model_entry[1]
            best_selected_hyperparams = results.get(best_model_name, {}).get('selected_hyperparameters', []) or []
            break

    final_hyperparams = {}
    if best_selected_hyperparams:
        tuples = [tuple(sorted(d.items())) for d in best_selected_hyperparams]
        most_common_tuple, _ = Counter(tuples).most_common(1)[0]
        final_hyperparams = dict(most_common_tuple)

    predictions = None
    if best_model_class is not None:
        try:
            model_instance = best_model_class(**final_hyperparams) if final_hyperparams else best_model_class()
            model_instance.fit(X_train, y_train)
            if X_test is not None:
                predictions = model_instance.predict(X_test)
        except Exception:
            predictions = None

    return best_model_name, results, predictions

if __name__ == "__main__":
    pass
