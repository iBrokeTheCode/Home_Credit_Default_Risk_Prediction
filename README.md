---
title: Home Credit Default Risk Prediction
emoji: 🍃
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
license: mit
short_description: ML Classification models applied to Home Credit Risk dataset
---

# 🏦 Home Credit Default Risk Prediction

## Table of Contents

1. [Project Description](#1-project-description)
2. [Methodology & Key Features](#2-methodology--key-features)
3. [Technology Stack](#3-technology-stack)
4. [Dataset](#4-dataset)

## 1. Project Description

This project focuses on building a machine learning pipeline to predict a client's ability to repay a loan. It is a binary classification task that uses a real-world financial dataset to identify clients who may face payment difficulties.

The project goes beyond a standard model by including a practical application that:

- **Preprocesses and cleans the dataset** for model training.
- **Trains a machine learning model** to predict loan repayment risk.
- **Deploys an interactive predictor app** using Marimo, hosted on Hugging Face Spaces.
- **Allows users to make predictions** by providing the top 10 most influential features.

This work showcases a complete end-to-end workflow, transforming raw data into a functional, user-friendly tool for risk assessment.

> [!IMPORTANT]
>
> - Check out the deployed app here: 👉️ [Home Credit Default Risk Prediction App](https://huggingface.co/spaces/iBrokeTheCode/Home_Credit_Default_Risk_Prediction) 👈️
> - Check out the Jupyter Notebook for a detailed walkthrough of the project here: 👉️ [Jupyter Notebook](https://huggingface.co/spaces/iBrokeTheCode/Home_Credit_Default_Risk_Prediction/blob/main/tutorial_app.ipynb) 👈️

![App](./public/app-demo.png)

## 2. Methodology & Key Features

- **Model Selection:** Four different models were trained and evaluated, with **LightGBM** selected as the final model due to its superior performance, achieving a **ROC AUC score of 0.751** on the test set.
- **Automated Preprocessing:** The data preprocessing pipeline handles common tasks such as feature scaling and categorical encoding, ensuring the model receives clean and formatted data.
- **Interactive Predictor:** An application built with **Marimo** allows users to interact with the trained model directly. It uses the **top 10 most important features**—identified from the final LightGBM model—to generate real-time predictions.

## 3. Technology Stack

This project was built using the following technologies and libraries:

**Dashboard & Hosting:**

- [Marimo](https://github.com/marimo-team/marimo): A Python library for building interactive dashboards.
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-config-reference): Used for hosting and sharing the interactive dashboard.

**Data Analysis & Visualization:**

- [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.
- [Matplotlib](https://matplotlib.org/): For creating static visualizations.
- [Seaborn](https://seaborn.pydata.org/): For creating statistical graphics.

**Modeling & Training:**

- [Scikit-Learn](https://scikit-learn.org/stable/): For machine learning tasks such as preprocessing, feature engineering, and model training.
- [LightGBM](https://lightgbm.readthedocs.io/en/stable/): It is a gradient boosting framework that uses tree based learning algorithms.

**Development Tools:**

- [Ruff](https://github.com/charliermarsh/ruff): A fast Python linter and code formatter.
- [uv](https://github.com/astral-sh/uv): A fast Python package installer and resolver.

## 4. Dataset

This project utilizes the **Home Credit Default Risk** from Kaggle, a public dataset containing details on over 246,000 of individuals who have made payments on their loans.

- **Source**: [Kaggle Dataset](https://www.kaggle.com/competitions/home-credit-default-risk/overview)
