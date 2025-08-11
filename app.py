import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import joblib
    import warnings

    import marimo as mo
    import pandas as pd

    warnings.filterwarnings(
        "ignore", message="X does not have valid feature names"
    )
    return joblib, mo, pd


@app.cell
def _(mo):
    mo.center(mo.md("# 🏦 Home Credit Default Risk Prediction"))
    return


@app.cell
def _(mo):
    mo.Html("<br><hr><br>")
    return


@app.cell
def _(joblib, mo):
    # 📌 [1] Load the saved model pipeline
    with mo.redirect_stdout():
        loaded_pipeline = joblib.load("./model/lgbm_model.joblib")
    return (loaded_pipeline,)


@app.cell
def _():
    # 📌 [2] Define the default values for all other features
    default_values = {
        "SK_ID_CURR": 277659.5,
        "CNT_CHILDREN": 0.0,
        "AMT_INCOME_TOTAL": 147150.0,
        "AMT_CREDIT": 512997.75,
        "AMT_ANNUITY": 24885.0,
        "AMT_GOODS_PRICE": 450000.0,
        "REGION_POPULATION_RELATIVE": 0.01885,
        "DAYS_BIRTH": -15743.5,
        "DAYS_EMPLOYED": -1219.0,
        "DAYS_REGISTRATION": -4492.0,
        "DAYS_ID_PUBLISH": -3254.0,
        "OWN_CAR_AGE": 9.0,
        "FLAG_MOBIL": 1.0,
        "FLAG_EMP_PHONE": 1.0,
        "FLAG_WORK_PHONE": 0.0,
        "FLAG_CONT_MOBILE": 1.0,
        "FLAG_PHONE": 0.0,
        "FLAG_EMAIL": 0.0,
        "CNT_FAM_MEMBERS": 2.0,
        "REGION_RATING_CLIENT": 2.0,
        "REGION_RATING_CLIENT_W_CITY": 2.0,
        "HOUR_APPR_PROCESS_START": 12.0,
        "REG_REGION_NOT_LIVE_REGION": 0.0,
        "REG_REGION_NOT_WORK_REGION": 0.0,
        "LIVE_REGION_NOT_WORK_REGION": 0.0,
        "REG_CITY_NOT_LIVE_CITY": 0.0,
        "REG_CITY_NOT_WORK_CITY": 0.0,
        "LIVE_CITY_NOT_WORK_CITY": 0.0,
        "EXT_SOURCE_1": 0.5068839442599388,
        "EXT_SOURCE_2": 0.5662837032261614,
        "EXT_SOURCE_3": 0.5370699579791587,
        "APARTMENTS_AVG": 0.0876,
        "BASEMENTAREA_AVG": 0.0764,
        "YEARS_BEGINEXPLUATATION_AVG": 0.9816,
        "YEARS_BUILD_AVG": 0.7552,
        "COMMONAREA_AVG": 0.0211,
        "ELEVATORS_AVG": 0.0,
        "ENTRANCES_AVG": 0.1379,
        "FLOORSMAX_AVG": 0.1667,
        "FLOORSMIN_AVG": 0.2083,
        "LANDAREA_AVG": 0.0483,
        "LIVINGAPARTMENTS_AVG": 0.0756,
        "LIVINGAREA_AVG": 0.0746,
        "NONLIVINGAPARTMENTS_AVG": 0.0,
        "NONLIVINGAREA_AVG": 0.0035,
        "APARTMENTS_MODE": 0.084,
        "BASEMENTAREA_MODE": 0.0748,
        "YEARS_BEGINEXPLUATATION_MODE": 0.9816,
        "YEARS_BUILD_MODE": 0.7648,
        "COMMONAREA_MODE": 0.0191,
        "ELEVATORS_MODE": 0.0,
        "ENTRANCES_MODE": 0.1379,
        "FLOORSMAX_MODE": 0.1667,
        "FLOORSMIN_MODE": 0.2083,
        "LANDAREA_MODE": 0.0459,
        "LIVINGAPARTMENTS_MODE": 0.0771,
        "LIVINGAREA_MODE": 0.0731,
        "NONLIVINGAPARTMENTS_MODE": 0.0,
        "NONLIVINGAREA_MODE": 0.0011,
        "APARTMENTS_MEDI": 0.0864,
        "BASEMENTAREA_MEDI": 0.0761,
        "YEARS_BEGINEXPLUATATION_MEDI": 0.9816,
        "YEARS_BUILD_MEDI": 0.7585,
        "COMMONAREA_MEDI": 0.0209,
        "ELEVATORS_MEDI": 0.0,
        "ENTRANCES_MEDI": 0.1379,
        "FLOORSMAX_MEDI": 0.1667,
        "FLOORSMIN_MEDI": 0.2083,
        "LANDAREA_MEDI": 0.0488,
        "LIVINGAPARTMENTS_MEDI": 0.0765,
        "LIVINGAREA_MEDI": 0.0749,
        "NONLIVINGAPARTMENTS_MEDI": 0.0,
        "NONLIVINGAREA_MEDI": 0.003,
        "TOTALAREA_MODE": 0.0687,
        "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_60_CNT_SOCIAL_CIRCLE": 0.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "DAYS_LAST_PHONE_CHANGE": -755.0,
        "FLAG_DOCUMENT_2": 0.0,
        "FLAG_DOCUMENT_3": 1.0,
        "FLAG_DOCUMENT_4": 0.0,
        "FLAG_DOCUMENT_5": 0.0,
        "FLAG_DOCUMENT_6": 0.0,
        "FLAG_DOCUMENT_7": 0.0,
        "FLAG_DOCUMENT_8": 0.0,
        "FLAG_DOCUMENT_9": 0.0,
        "FLAG_DOCUMENT_10": 0.0,
        "FLAG_DOCUMENT_11": 0.0,
        "FLAG_DOCUMENT_12": 0.0,
        "FLAG_DOCUMENT_13": 0.0,
        "FLAG_DOCUMENT_14": 0.0,
        "FLAG_DOCUMENT_15": 0.0,
        "FLAG_DOCUMENT_16": 0.0,
        "FLAG_DOCUMENT_17": 0.0,
        "FLAG_DOCUMENT_18": 0.0,
        "FLAG_DOCUMENT_19": 0.0,
        "FLAG_DOCUMENT_20": 0.0,
        "FLAG_DOCUMENT_21": 0.0,
        "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Laborers",
        "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "FONDKAPREMONT_MODE": "reg oper account",
        "HOUSETYPE_MODE": "block of flats",
        "WALLSMATERIAL_MODE": "Panel",
        "EMERGENCYSTATE_MODE": "No",
    }
    return (default_values,)


@app.cell
def _(mo):
    # 📌 [3] Create widgets for the top 10 features
    EXT_SOURCE_3 = mo.ui.slider(
        start=0.00,
        stop=0.90,
        step=0.01,
        value=0.5,
        label="EXT_SOURCE_3",
    )

    EXT_SOURCE_2 = mo.ui.slider(
        start=0.00,
        stop=0.86,
        step=0.01,
        value=0.5,
        label="EXT_SOURCE_2",
    )

    DAYS_BIRTH = mo.ui.slider(
        start=-25229,
        stop=-7673,
        value=-15743,
        label="DAYS_BIRTH",
    )

    EXT_SOURCE_1 = mo.ui.slider(
        start=0.01,
        stop=0.97,
        step=0.01,
        value=0.5,
        label="EXT_SOURCE_1",
    )

    AMT_ANNUITY = mo.ui.slider(
        start=1980,
        stop=258025,
        step=100,
        value=24885,
        label="AMT_ANNUITY",
    )

    AMT_CREDIT = mo.ui.slider(
        start=45000,
        stop=4050000,
        step=50000,
        value=512997,
        label="AMT_CREDIT",
    )

    DAYS_EMPLOYED = mo.ui.slider(
        start=-17583,
        stop=365243,
        value=-1219,
        label="DAYS_EMPLOYED",
    )

    DAYS_ID_PUBLISH = mo.ui.slider(
        start=-7197,
        stop=0,
        value=-3254,
        label="DAYS_ID_PUBLISH",
    )

    DAYS_REGISTRATION = mo.ui.slider(
        start=-24672,
        stop=0,
        value=-4492,
        label="DAYS_REGISTRATION",
    )

    SK_ID_CURR = mo.ui.slider(
        start=100003,
        stop=456253,
        step=100,
        value=277659,
        label="SK_ID_CURR",
    )

    features_widgets = {
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "DAYS_BIRTH": DAYS_BIRTH,
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_CREDIT": AMT_CREDIT,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
        "DAYS_REGISTRATION": DAYS_REGISTRATION,
        "SK_ID_CURR": SK_ID_CURR,
    }
    return (features_widgets,)


@app.cell
def _(features_widgets, mo):
    # 📌 [4] Create the form with the sliders
    sliders_form = (
        mo.md("""
        ### Enter Client Information

        {EXT_SOURCE_3}
        {EXT_SOURCE_2}
        {DAYS_BIRTH}
        {EXT_SOURCE_1}
        {AMT_ANNUITY}
        {AMT_CREDIT}
        {DAYS_EMPLOYED}
        {DAYS_ID_PUBLISH}
        {DAYS_REGISTRATION}
        {SK_ID_CURR}
        """)
        .batch(**features_widgets)  # Pass the dict unpacked
        .form(show_clear_button=True, bordered=True)
    )
    return (sliders_form,)


@app.cell
def _(sliders_form):
    # 📌 [5] Display the form
    sliders_form
    return


@app.cell
def _(default_values, loaded_pipeline, mo, pd, sliders_form):
    # 📌 [6] Get prediction from model
    probability = None

    # Process form submission
    if sliders_form.value is not None:
        # Copy default values
        prediction_data = default_values.copy()

        # Update with sliders' submitted values
        prediction_data.update(sliders_form.value)

        # Create a DataFrame
        predict_df = pd.DataFrame([prediction_data])

        # Predict probability
        probability = loaded_pipeline.predict_proba(predict_df)[:, 1][0]
    else:
        mo.md("Fill in the form and click **Submit** to get a prediction.")
    return (probability,)


@app.cell
def _(probability):
    # 📌 [7] Display prediction results
    prob_percent = 70.12
    risk = "High Risk"
    direction = "decrease"

    if probability is not None:
        prob_percent = round(probability * 100, 2)

        # Define risk category
        if probability < 0.34:
            risk = "Low Risk"
            direction = "increase"
        elif probability < 0.67:
            risk = "Medium Risk"
            direction = None
        else:
            risk = "High Risk"
            direction = "decrease"
    return direction, prob_percent, risk


@app.cell
def _(mo):
    mo.Html("<br>")
    return


@app.cell
def _(mo):
    mo.md("## 🔮 Credit Risk Prediction")
    return


@app.cell
def _(mo):
    mo.Html("<hr><br>")
    return


@app.cell
def _(direction, mo, prob_percent, risk):
    interpretation_text = f"""This means there is a {prob_percent}% chance the client will **default on their loan**.  
    Risk level is categorized as **{risk}**, which can help guide loan approval decisions.
    """

    result_stat = mo.stat(
        label="🎲 Probability of Payment Difficulties",
        bordered=True,
        value=f"{prob_percent}%",
        caption=risk,
        direction=direction,
    )

    interpretation_stat = mo.stat(
        label="💡 Interpretation",
        bordered=True,
        value="",
        caption=interpretation_text,
    )
    return interpretation_stat, result_stat


@app.cell
def _(interpretation_stat, mo, result_stat):
    mo.vstack(
        items=[
            mo.hstack(
                items=[result_stat, interpretation_stat], widths="equal", gap=1
            ),
        ],
        gap=1,
        heights="equal",
    )
    return


@app.cell
def _(mo):
    mo.Html("<br><hr>")
    return


@app.cell
def _(mo):
    mo.callout(
        kind="info",
        value=mo.md(
            """💡 **Want a step-by-step walkthrough instead?**   
        Check the Jupyter notebook version here: 👉 [Jupyter notebook](https://huggingface.co/spaces/iBrokeTheCode/Home_Credit_Default_Risk_Prediction/blob/main/tutorial_app.ipynb)""",
        ),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 🚀 Model Selection""")
    return


@app.cell
def _(mo):
    mo.Html("<hr><br>")
    return


@app.cell
def _(mo):
    lg_stat = mo.stat(
        label="Logistic Regression",
        bordered=True,
        value="💪🏻 0.687 📝 0.685",
        caption="Scores are consistent across train and test, indicating no overfitting. However, the overall AUC is low, suggesting underfitting — the model is too simple to capture complex patterns.",
        direction="decrease",
    )

    rfc_stat = mo.stat(
        label="Random Forest Classifier",
        bordered=True,
        value="💪🏻 1.0 📝 0.707",
        caption="Perfect training AUC indicates severe overfitting — the model memorized the training set. While the test score is better than Logistic Regression, the gap is too large for good generalization.",
        direction="decrease",
    )

    rfo_stat = mo.stat(
        label="Random Forest with Randomized Search",
        bordered=True,
        value="💪🏻 0.820 📝 0.731",
        caption="Hyperparameter tuning greatly reduced overfitting. The smaller train–test gap and improved test AUC show better generalization and a strong performance.",
        direction="increase",
    )

    lgbm_stat = mo.stat(
        label="LightGBM",
        bordered=True,
        value="💪🏻 0.852 📝 0.751",
        caption="Best overall performance. Small train–test gap and highest test AUC indicate a well-balanced model with strong generalization.",
        direction="increase",
    )

    mo.vstack(
        items=[
            mo.hstack(items=[lg_stat, rfc_stat], widths="equal", gap=1),
            mo.hstack(items=[rfo_stat, lgbm_stat], widths="equal", gap=1),
        ],
        gap=1,
        heights="equal",
        align="center",
        justify="center",
    )
    return


@app.cell
def _(mo):
    mo.Html("<br>")
    return


@app.cell
def _(mo):
    mo.md(
        r"""Based on a comparison of all the models _(using AUC ROC metric)_, the final model selection is clear."""
    )
    return


@app.cell
def _(mo):
    mo.Html("<br>")
    return


@app.cell
def _(mo):
    mo.center(
        mo.md(r"""
    | Model | 💪🏻 Train Score | 📝 Test Score |
    | :--- | :---: | :---: |
    | Logistic Regression | 0.687 | 0.685 |
    | Random Forest Classifier | 1.000 | 0.707 |
    | Randomized Search (Tuned RF) | 0.820 | 0.731 |
    | **LightGBM** | 0.852 | **0.751** |
    """)
    )
    return


@app.cell
def _(mo):
    mo.Html("<br>")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    * The **Logistic Regression** model performed poorly due to underfitting.
    * The base **Random Forest** model, while better, suffered from severe overfitting.
    * The tuned **Random Forest** model was a significant improvement and a strong contender, achieving a solid `test_score`.
    * However, the **LightGBM** model ultimately demonstrated the best performance, achieving the highest **ROC AUC test score of 0.751**. This indicates that it is the most robust and accurate model for predicting loan repayment risk on unseen data.
    """
    )
    return


@app.cell
def _(mo):
    mo.Html("<br><hr><br>")
    return


@app.cell
def _(mo):
    mo.center(
        mo.md(
            "**Connect with me:** 💼 [Linkedin](https://www.linkedin.com/in/alex-turpo/) • 🐱 [GitHub](https://github.com/iBrokeTheCode) • 🤗 [Hugging Face](https://huggingface.co/iBrokeTheCode)"
        )
    )
    return


if __name__ == "__main__":
    app.run()
