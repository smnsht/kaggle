import pandas as pd
import numpy as np


def load_raw_data(path):
    return pd.read_csv(path)


def add_engineered_features(df):
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    porch_cols = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
    ]
    df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)

    return df


def add_neighborhood_median(df, target_series):
    medians = target_series.groupby(df["Neighborhood"]).median()
    df["NeighborhoodMedian"] = df["Neighborhood"].map(medians)
    return df


def impute_missing_values(df):
    df = df.copy()

    # Numeric columns → median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns → "Missing"
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Missing")

    return df


def one_hot_encode(df):
    return pd.get_dummies(df, drop_first=True)


def log_target(y):
    return np.log1p(y)
