from numpy import nan, ndarray
from pandas import DataFrame, concat
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(train_df: DataFrame, test_df: DataFrame) -> tuple[ndarray, ndarray]:
    """
    Pre process data for modeling. Receives train and test dataframes, cleans them up, and returns ndarrays with feature engineering already performed.

    Args:
        train_df (DataFrame): The training dataframe.
        test_df (DataFrame): The test dataframe.

    Returns:
        tuple[ndarray, ndarray]: A tuple with the preprocessed train and test data as ndarrays
    """
    aux_train_df = train_df.copy()
    aux_test_df = test_df.copy()

    # 📌 [1] Correct outliers/anomalous values in numerical columns
    aux_train_df["DAYS_EMPLOYED"] = aux_train_df["DAYS_EMPLOYED"].replace({365243: nan})
    aux_test_df["DAYS_EMPLOYED"] = aux_test_df["DAYS_EMPLOYED"].replace({365243: nan})

    # 📌 [2] Encode string categorical features
    categorical_cols = aux_train_df.select_dtypes(include="object").columns
    binary_cols = [col for col in categorical_cols if aux_train_df[col].nunique() == 2]
    multi_cols = [col for col in categorical_cols if aux_train_df[col].nunique() > 2]

    # [2.1] Encode Binary Categorical Features
    ordinal_encoder = OrdinalEncoder()

    ordinal_encoder.fit(aux_train_df[binary_cols])
    aux_train_df[binary_cols] = ordinal_encoder.transform(aux_train_df[binary_cols])
    aux_test_df[binary_cols] = ordinal_encoder.transform(aux_test_df[binary_cols])

    # [2.2] Encode Multi Categorical Features
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",  # Prevents errors when test set contain categories that didn't appear in train dataframe
        sparse_output=False,  # Returns a dense array instead of a sparse matrix
    )

    one_hot_encoder.fit(aux_train_df[multi_cols])
    ohe_train = one_hot_encoder.transform(aux_train_df[multi_cols])
    ohe_test = one_hot_encoder.transform(aux_test_df[multi_cols])

    # Get columns names
    ohe_cols = one_hot_encoder.get_feature_names_out(input_features=multi_cols)

    # Convert arrays to DataFrames
    ohe_train_df = DataFrame(data=ohe_train, columns=ohe_cols, index=aux_train_df.index)  # type: ignore
    ohe_test_df = DataFrame(data=ohe_test, columns=ohe_cols, index=aux_test_df.index)  # type: ignore

    # Drop original multi category columns
    aux_train_df.drop(columns=multi_cols, inplace=True)
    aux_test_df.drop(columns=multi_cols, inplace=True)

    # Concatenate encoded dataframe
    aux_train_df = concat([aux_train_df, ohe_train_df], axis=1)
    aux_test_df = concat([aux_test_df, ohe_test_df], axis=1)

    # 📌 [3] Impute values for columns with missing data
    imputer = SimpleImputer(strategy="median")
    imputer.fit(aux_train_df)

    imputer_train = imputer.transform(aux_train_df)
    imputer_test = imputer.transform(aux_test_df)

    aux_train_df = DataFrame(
        data=imputer_train,  # type: ignore
        columns=aux_train_df.columns,
        index=aux_train_df.index,
    )
    aux_test_df = DataFrame(
        data=imputer_test,  # type: ignore
        columns=aux_test_df.columns,
        index=aux_test_df.index,
    )

    # 📌 [4]  Feature Scaling with Min-Max Scaler
    scaler = MinMaxScaler()
    scaler.fit(aux_train_df)

    scaler_train = scaler.transform(aux_train_df)
    scaler_test = scaler.transform(aux_test_df)

    return scaler_train, scaler_test


def preprocess_data_pipeline(
    train_df: DataFrame, test_df: DataFrame
) -> tuple[ndarray | spmatrix, ndarray | spmatrix]:
    """
    Pre process data for modeling. Receives train and test dataframes, cleans them up, and returns ndarrays with feature engineering already performed.

    Args:
        train_df (DataFrame): The training dataframe.
        test_df (DataFrame): The test dataframe.

    Returns:
        tuple[ndarray, ndarray]: A tuple with the preprocessed train and test data as ndarrays
    """
    # Create copies to avoid modifying original dataframes
    aux_train_df = train_df.copy()
    aux_test_df = test_df.copy()

    # 📌 [1] Correct outliers/anomalous values in numerical columns
    aux_train_df["DAYS_EMPLOYED"] = aux_train_df["DAYS_EMPLOYED"].replace({365243: nan})
    aux_test_df["DAYS_EMPLOYED"] = aux_test_df["DAYS_EMPLOYED"].replace({365243: nan})

    # 📌 [2] Define column types for the ColumnTransformer
    numerical_cols = aux_train_df.select_dtypes(include="number").columns.to_list()
    categorical_cols = aux_train_df.select_dtypes(include="object").columns.to_list()

    binary_cols = [col for col in categorical_cols if aux_train_df[col].nunique() == 2]
    multi_cols = [col for col in categorical_cols if aux_train_df[col].nunique() > 2]

    # 📌 [3] Build the preprocessing pipeline using ColumnTransformer
    # Create a pipeline for numerical columns: Impute and Scale processes
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder()),
            ("scaler", MinMaxScaler()),
        ]
    )

    multi_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Create a ColumnTransformer object with the defined pipelines and transformers
    preprocessor = ColumnTransformer(
        transformers=[
            # Tuple format: ('name', transformer, list_of_columns)
            ("binary", binary_pipeline, binary_cols),
            ("multi", multi_pipeline, multi_cols),
            ("numerical", numerical_pipeline, numerical_cols),
        ],
        remainder="passthrough",
    )

    # 📌 [4] Fit and transform the data
    preprocessor.fit(aux_train_df)
    train_preprocessed = preprocessor.transform(aux_train_df)
    test_preprocessed = preprocessor.transform(aux_test_df)

    return train_preprocessed, test_preprocessed
