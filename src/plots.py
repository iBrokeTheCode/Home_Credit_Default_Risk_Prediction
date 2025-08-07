import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame, Series

from src.theme import custom_palette


def plot_target_distribution(df: DataFrame) -> tuple[DataFrame, Figure]:
    """
    Plot the distribution of the 'TARGET' column in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing the 'TARGET' column.

    Returns:
        DataFrame: A DataFrame containing the count and percentage of each class.
        Figure: The matplotlib Figure object containing the plot.
    """
    target_counts = df["TARGET"].value_counts()
    target_percent = (target_counts / target_counts.sum() * 100).round(2)

    # Combine into a DataFrame for clarity
    target_df = target_counts.to_frame(name="Count")
    target_df["Percentage"] = target_percent

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=target_df,
        x="TARGET",
        y="Count",
        hue="TARGET",
        palette=custom_palette[:2],
    )

    # Titles and formatting
    ax.set_xlabel("Payment Difficulties (1 = Yes, 0 = No)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    return target_df, fig


def plot_credit_amounts(df: DataFrame) -> Figure:
    """
    Plot a histogram of credit amounts.

    Args:
        df (DataFrame): The DataFrame containing the credit amount data.

    Returns:
        Figure: The matplotlib figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x="AMT_CREDIT", bins=100, kde=True, color=custom_palette[0])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def plot_education_levels(df: DataFrame) -> tuple[DataFrame, Figure]:
    """
    Plot a bar chart of education levels.

    Args:
        df (DataFrame): The DataFrame containing the education level data.

    Returns:
        DataFrame: The DataFrame containing the education level counts and percentages.
        Figure: The matplotlib figure object containing the plot.
    """
    education_count = (
        df["NAME_EDUCATION_TYPE"].value_counts().sort_values(ascending=False)
    )
    education_percentage = (education_count / df.shape[0] * 100).round(2)

    education_df = education_count.to_frame(name="Count")
    education_df["Percentage"] = education_percentage

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        y="NAME_EDUCATION_TYPE",
        hue="NAME_EDUCATION_TYPE",
        palette=custom_palette[:5],
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("Education Level")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return education_df, fig


def plot_occupation(df: DataFrame) -> tuple[Series, Figure]:
    """
    Plot the distribution of occupations in the dataset.

    Args:
        df (DataFrame): The DataFrame containing the data.

    Returns:
        Series: A Series containing the count of each occupation.
        Figure: A Matplotlib Figure object containing the plot.
    """
    occupation_df = df["OCCUPATION_TYPE"].value_counts(dropna=False, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=occupation_df.values,
        y=occupation_df.index,
        hue=occupation_df.index,
        legend=False,
    )
    ax.set_xlabel("Number of Applicants")
    ax.set_ylabel("Occupation")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return occupation_df, fig


def plot_family_status(df: DataFrame) -> tuple[Series, Figure]:
    """
    Plot the distribution of family statuses in the dataset.

    Args:
        df (DataFrame): The DataFrame containing the data.

    Returns:
        Series: A Series containing the count of each family status.
        Figure: A Matplotlib Figure object containing the plot.
    """
    family_status_df = df["NAME_FAMILY_STATUS"].value_counts(
        dropna=False, ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=family_status_df.values,
        y=family_status_df.index,
        hue=family_status_df.index,
        palette=custom_palette[:6],
        legend=False,
    )
    ax.set_xlabel("Number of Applicants")
    ax.set_ylabel("Family Status")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return family_status_df, fig


def plot_income_type(df: DataFrame) -> Figure:
    """
    Plot the count of income types for each target group.

    Args:
        df (DataFrame): The DataFrame containing the data.

    Returns:
        Figure: A Matplotlib Figure object containing the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df, y="NAME_INCOME_TYPE", hue="TARGET", palette=custom_palette[:2]
    )
    ax1.legend(loc="lower right", title="Target")
    ax1.set_xlabel("Number of Applicants")
    ax1.set_ylabel("Income Type")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig
