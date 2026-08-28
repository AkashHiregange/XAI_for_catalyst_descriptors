from pathlib import Path

import pandas as pd


def load_data(features_path,properties_path,features_columns_drop=None,properties_columns_drop=None):
    """Load feature and target tables from CSV or Excel files."""

    def read_table(path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)

        raise ValueError(f"Unsupported file format: {suffix}")

    features = read_table(features_path)
    properties = read_table(properties_path)

    if features_columns_drop:
        features = features.drop(columns=features_columns_drop)

    if properties_columns_drop:
        properties = properties.drop(columns=properties_columns_drop)

    return features, properties