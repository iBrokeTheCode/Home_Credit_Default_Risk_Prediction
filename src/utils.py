from pandas import DataFrame, Series, read_csv
from sklearn.model_selection import train_test_split

from src.config import DATASET_FILE_PATH


def get_dataset() -> DataFrame:
    """
    Get the dataset

    Returns:
        DataFrame: The dataset as a DataFrame
    """
    try:
        return DataFrame(data=read_csv(DATASET_FILE_PATH))
    except FileNotFoundError:
        return DataFrame(data={})


def get_features_target(df: DataFrame) -> tuple[DataFrame, Series]:
    """
    Get the feature and target from the dataset

    Args:
        df (DataFrame): The dataset as a DataFrame

    Returns:
        tuple[DataFrame, Series]: The features and target as a tuple
    """
    return df.drop(columns=["TARGET"], axis=1), df["TARGET"]


def get_train_test_sets(
    X: DataFrame, y: Series
) -> tuple[DataFrame, Series, DataFrame, Series]:
    """
    Get the train and test sets from the features and target

    Args:
        features (DataFrame): The features as a DataFrame
        target (Series): The target as a Series

    Returns:
        tuple[DataFrame, Series, DataFrame, Series]: The train and test sets as a tuple
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test
