import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.center(mo.md("# Home Credit Default Risk Prediction"))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    from src.plots import (
        plot_target_distribution,
        plot_credit_amounts,
        plot_education_levels,
        plot_occupation,
        plot_family_status,
        plot_income_type,
    )
    from src.theme import custom_palette
    from src.utils import get_dataset, get_features_target, get_train_test_sets
    from src.preprocessing import preprocess_data_pipeline
    return (
        LogisticRegression,
        RandomForestClassifier,
        get_dataset,
        get_features_target,
        get_train_test_sets,
        pd,
        plot_credit_amounts,
        plot_education_levels,
        plot_family_status,
        plot_income_type,
        plot_occupation,
        plot_target_distribution,
        preprocess_data_pipeline,
        roc_auc_score,
    )


@app.cell
def _(get_dataset, get_features_target):
    df = get_dataset()
    X, y = get_features_target(df)
    return X, df, y


@app.cell
def _(mo):
    mo.md("""## 1. Exploratory Data Analysis""")
    return


@app.cell
def _(mo):
    mo.md("""### 1.1 Dataset Information""")
    return


@app.cell
def _(mo):
    mo.md("""**a. Shape of the train and test datasets**""")
    return


@app.cell
def _(X_test, X_train, df):
    train_samples = "Train dataset samples: {}".format(X_train.shape[0])
    test_samples = "Test dataset samples: {}".format(X_test.shape[0])
    columns_number = "Number of columns: {}".format(df.shape[1])

    train_samples, test_samples, columns_number
    return


@app.cell
def _(mo):
    mo.md("""**b. Dataset features**""")
    return


@app.cell
def _(X):
    X.columns
    return


@app.cell
def _(mo):
    mo.md("""**c. Sample from dataset**""")
    return


@app.cell
def _(X):
    sample = X.head(5).T
    sample.columns = [str(col) for col in sample.columns]
    sample
    return


@app.cell
def _(mo):
    mo.md("""**d. Target variable Distribution**""")
    return


@app.cell
def _(df, plot_target_distribution):
    target_table, target_plot = plot_target_distribution(df=df)
    target_table
    return (target_plot,)


@app.cell
def _(target_plot):
    target_plot
    return


@app.cell
def _(mo):
    mo.md("""**e. Number of columns of each data type**""")
    return


@app.cell
def _(X):
    X.dtypes.value_counts().sort_values(ascending=False)
    return


@app.cell
def _(X):
    categorical_cols = (
        X.select_dtypes(include=["object"]).nunique().sort_values(ascending=False)
    )
    categorical_cols
    return


@app.cell
def _(mo):
    mo.md("""**f. Missing data**""")
    return


@app.cell
def _(X, pd):
    missing_count = X.isna().sum().sort_values(ascending=False)
    missing_percentage = (missing_count / X.shape[0] * 100).round(2)

    missing_data = pd.DataFrame(
        data={"Count": missing_count, "percentage": missing_percentage}
    )
    missing_data
    return


@app.cell
def _(mo):
    mo.md("""### 1.2 Distribution of Variables""")
    return


@app.cell
def _(mo):
    mo.md("""**a. Credit Amounts**""")
    return


@app.cell
def _(X, plot_credit_amounts):
    plot_credit_amounts(df=X)
    return


@app.cell
def _(mo):
    mo.md("""**b. Education Level of Credit Applicants**""")
    return


@app.cell
def _(X, plot_education_levels):
    education_table, education_plot = plot_education_levels(df=X)
    education_table
    return (education_plot,)


@app.cell
def _(education_plot):
    education_plot
    return


@app.cell
def _(mo):
    mo.md("""**c. Ocupation of Credit Applicants**""")
    return


@app.cell
def _(X, plot_occupation):
    occupation_table, occupation_plot = plot_occupation(df=X)
    occupation_table
    return (occupation_plot,)


@app.cell
def _(occupation_plot):
    occupation_plot
    return


@app.cell
def _(mo):
    mo.md("""**d. Family Status of Applicants**""")
    return


@app.cell
def _(X, plot_family_status):
    family_status_table, family_status_plot = plot_family_status(df=X)
    family_status_table
    return (family_status_plot,)


@app.cell
def _(family_status_plot):
    family_status_plot
    return


@app.cell
def _(mo):
    mo.md("""**e. Income Type of Applicants by Target Variable**""")
    return


@app.cell
def _(df, plot_income_type):
    plot_income_type(df=df)
    return


@app.cell
def _(mo):
    mo.md("""## 2. Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md("""**a. Separate Train and Test Datasets**""")
    return


@app.cell
def _(X, get_train_test_sets, y):
    X_train, y_train, X_test, y_test = get_train_test_sets(X, y)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md("""**b. Preprocess Data**""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This preprocessing perform:

    - Correct outliers/anomalous values in numerical columns (`DAYS_EMPLOYED` column).
    - Encode string categorical features (`dtype object`).
        - If the feature has 2 categories, Binary Encoding is applied.
        - One Hot Encoding for more than 2 categories.
    - Impute values for all columns with missing data (using median as imputing value).
    - Feature scaling with Min-Max scaler
    """
    )
    return


@app.cell
def _(X_test, X_train, preprocess_data_pipeline):
    train_data, test_data = preprocess_data_pipeline(
        train_df=X_train, test_df=X_test
    )
    train_data.shape, test_data.shape
    return test_data, train_data


@app.cell
def _(mo):
    mo.md("""## 3. Training Models""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""At this points, we will work with `train_data` and `test_data` as features sets; also `y_train` and `y_test` as target sets."""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### 3.1 Logistic Regression""")
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("""
    In Logistic Regression, C is the inverse of regularization strength:

    - **Small C** → Stronger regularization → Simpler model, less overfitting risk, but may underfit.
    - **Large C** → Weaker regularization → Model fits training data more closely, but may overfit.
    """),
        kind="info",
    )
    return


@app.cell
def _(
    LogisticRegression,
    roc_auc_score,
    test_data,
    train_data,
    y_test,
    y_train,
):
    # 📌 Logistic Regression
    log_reg = LogisticRegression(C=0.0001)
    log_reg.fit(train_data, y_train)

    # Train data predicton (class 1)
    lr_train_pred = log_reg.predict_proba(train_data)[:, 1]

    # Test data prediction (class 1)
    lr_test_pred = log_reg.predict_proba(test_data)[:, 1]

    # Get the ROC AUC Score on train and test datasets
    log_reg_scores = {
        "train_score": roc_auc_score(y_train, lr_train_pred),
        "test_score": roc_auc_score(y_test, lr_test_pred),
    }
    log_reg_scores
    return


@app.cell
def _(mo):
    mo.md(r"""### 3.2 Random Forest Classifier""")
    return


@app.cell
def _(
    RandomForestClassifier,
    roc_auc_score,
    test_data,
    train_data,
    y_test,
    y_train,
):
    # 📌 Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(train_data, y_train)

    rf_train_pred = rf.predict_proba(train_data)[:, 1]
    rf_test_pred = rf.predict_proba(test_data)[:, 1]

    rf_scores = {
        "train_score": roc_auc_score(y_train, rf_train_pred),
        "test_score": roc_auc_score(y_test, rf_test_pred),
    }
    rf_scores
    return


@app.cell
def _(mo):
    mo.md(r"""### 3.3. Randomized Search with Cross Validations""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We use this code snippet to use `RandomizedSearchCV`:

    ```py
    param_dist = {"n_estimators": [50, 100, 150], "max_depth": [10, 20, 30]}

    rf_optimized = RandomForestClassifier(random_state=42, n_jobs=-1)
    rscv = RandomizedSearchCV(
        estimator=rf_optimized,
        param_distributions=param_dist,
        n_iter=5,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    rscv.fit(train_data, y_train)

    rfo_train_pred = rscv.predict_proba(train_data)[:, 1]
    rfo_test_pred = rscv.predict_proba(test_data)[:, 1]

    rfo_scores = {
        "train_score": roc_auc_score(y_train, rfo_train_pred),
        "test_score": roc_auc_score(y_test, rfo_test_pred),
    }
    rfo_scores
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""📈 The obtained scores are:""")
    return


@app.cell
def _():
    rfo_scores = {
        "train_score": 0.820563139010308,
        "test_score": 0.7304320776838898,
    }
    rfo_scores
    return


@app.cell
def _(mo):
    mo.md(r"""🥇The best results are:""")
    return


@app.cell
def _(RandomForestClassifier):
    optimized_results = {
        "best_params_": {"n_estimators": 100, "max_depth": 10},
        "best_score_": 0.7296259755147781,
        "best_estimator_": RandomForestClassifier(
            max_depth=10, n_jobs=-1, random_state=42
        ),
    }
    optimized_results
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
