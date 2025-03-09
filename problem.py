import rampwf as rw
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

problem_title = 'Wine Competition'

# The original string region names
_prediction_label_names = np.array([
    'Bordeaux', 'Languedoc-Roussillon', 'Alsace', 'Champagne',
    'Vallee de la Loire', 'Bourgogne', 'Vallee du Rhone', 'Provence',
    'Sud Ouest', 'Corse', 'Savoie'
], dtype=object)

_target_column_name = 'region'

# Mapping from region names to integer labels
_label_to_index = {
    label: index for index, label in enumerate(_prediction_label_names)
}
_index_to_label = {
    index: label for index, label in enumerate(_prediction_label_names)
}

# Build Predictions using integer labels
_base_predictions = rw.prediction_types.make_multiclass(
    label_names=list(range(len(_prediction_label_names)))
)


# Override the Predictions class to ensure its label_names are integers.
class MyPredictions(_base_predictions):
    # Override the class attribute to be a list of integers.
    label_names = list(range(len(_prediction_label_names)))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


Predictions = MyPredictions

# Define the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.BalancedAccuracy(name='accuracy', precision=4),
]


def encode_label(y: pd.Series) -> pd.Series:
    """
    Encode a series of region names (strings) to their corresponding int
    labels.
    """
    return y.map(_label_to_index)


def decode_label(y: pd.Series) -> pd.Series:
    """
    Decode a series of integer labels to their region names.
    """
    return y.map(_index_to_label)


def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)
    return cv.split(X, y, groups=X['wine_id'])


def load_data(path='.', file='data_vivino_train.tsv'):
    path = Path(path) / "data"
    X_df = pd.read_csv(path / file, sep='\t')

    y = X_df[_target_column_name]
    X_df = X_df.drop(columns=[_target_column_name])

    # Encode region names to integers so they match Predictions.label_names.
    y = encode_label(y)
    return X_df, y


def get_train_data(path='.'):
    file = 'data_vivino_train.tsv'
    return load_data(path, file)


def get_test_data(path='.'):
    file = 'data_vivino_test.tsv'
    return load_data(path, file)
