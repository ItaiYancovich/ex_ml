print("sups")

from preprocess_faces import prepare_dataset


X, y, pipeline = prepare_dataset(
    folder="Train Set (Labeled)",
    fit_limit=800,
    n_components=225,
)
print(y[:10])