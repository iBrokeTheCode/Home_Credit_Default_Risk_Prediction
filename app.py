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
def _(mo):
    mo.md("""## Importing Libraries""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from src.utils import get_dataset, get_features_target, get_train_test_sets
    from src.theme import custom_palette
    return (
        custom_palette,
        get_dataset,
        get_features_target,
        get_train_test_sets,
        pd,
        plt,
        sns,
    )


@app.cell
def _(get_dataset, get_features_target, get_train_test_sets):
    df = get_dataset()
    X, y = get_features_target(df)
    X_train, y_train, X_test, y_test = get_train_test_sets(X, y)
    return X, X_test, X_train, df


@app.cell
def _(mo):
    mo.md("""## Exploratory Data Analysis""")
    return


@app.cell
def _(mo):
    mo.md("""### Dataset Information""")
    return


@app.cell
def _(mo):
    mo.md("""**a. Shape of the train and test datasets**""")
    return


@app.cell
def _(X_test, X_train, df):
    print("Train dataset samples: {}".format(X_train.shape[0]))
    print("Test dataset samples: {}".format(X_test.shape[0]))
    print("Number of columns: {}".format(df.shape[1]))
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
def _(custom_palette, df, plt, sns):
    # Get value counts and percentages
    target_counts = df["TARGET"].value_counts()
    target_percent = (target_counts / target_counts.sum() * 100).round(2)

    # Combine into a DataFrame for clarity
    target_df = target_counts.to_frame(name="Count")
    target_df["Percentage"] = target_percent

    # Plot
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=target_df,
        x="TARGET",
        y="Count",
        hue="TARGET",
        palette=custom_palette[:2],
    )

    # Titles and formatting
    plt.title("Distribution of TARGET variable")
    plt.xlabel("Payment Difficulties (1 = Yes, 0 = No)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    target_df
    return


@app.cell
def _(mo):
    mo.md("**e. Number of columns of each data type**")
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
    mo.md("**f. Missing data**")
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
    mo.md("### Distribution of Variables")
    return


@app.cell
def _(mo):
    mo.callout(kind="info", value="Continues at point 1.9")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
