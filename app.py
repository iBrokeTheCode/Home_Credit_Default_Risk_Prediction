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
    from src.utils import get_dataset, get_features_target, get_train_test_sets
    return get_dataset, get_features_target, get_train_test_sets


@app.cell
def _(get_dataset):
    df = get_dataset()
    return (df,)


@app.cell
def _(df, get_features_target):
    X, y = get_features_target(df)
    return X, y


@app.cell
def _(X, get_train_test_sets, y):
    X_train, y_train, X_test, y_test = get_train_test_sets(X, y)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, y_test, y_train):
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    return


if __name__ == "__main__":
    app.run()
