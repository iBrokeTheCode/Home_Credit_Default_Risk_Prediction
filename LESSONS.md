# Lessons

## Table of Contents

1. [🏗️ Building a Consistent Workflow with Pipelines and ColumnTransformers](#1-building-a-consistent-workflow-with-pipelines-and-columntransformers)
2. [🤖 Efficient Hyperparameter Tuning with RandomizedSearchCV](#2-efficient-hyperparameter-tuning-with-randomizedsearchcv)
3. [🚀 High-Performance Modeling with LightGBM](#3-high-performance-modeling-with-lightgbm)
4. [💾 Saving and Deploying a Complete Model Pipeline](#4-saving-and-deploying-a-complete-model-pipeline)

---

## 1. 🏗️ Building a Consistent Workflow with Pipelines and ColumnTransformers

A machine learning model is more than just an algorithm; it's a complete data processing workflow. The `Pipeline` and `ColumnTransformer` classes from `scikit-learn` are essential for creating a robust and reproducible process.

- `ColumnTransformer` allows you to apply different preprocessing steps (like scaling numerical data and encoding categorical data) to different columns in your dataset simultaneously.
- `Pipeline` chains these preprocessing steps with a final model. This ensures that the exact same transformations are applied to your data during training and prediction, preventing data leakage and consistency errors.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define different preprocessing steps for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor that applies these pipelines to the correct columns
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Build a final pipeline with the preprocessor and the model
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MyClassifier())
])

final_pipeline.fit(X_train, y_train)
```

---

## 2\. 🤖 Efficient Hyperparameter Tuning with RandomizedSearchCV

Hyperparameters are settings that are not learned from data but are set before training. Finding the best combination of these settings is crucial for optimal model performance.

- `RandomizedSearchCV` is a powerful and efficient method for hyperparameter tuning. Instead of exhaustively checking every possible combination like `GridSearchCV`, it samples a fixed number of combinations from a defined parameter space.
- This approach is much faster than an exhaustive search and often finds a very good set of hyperparameters, making it an excellent choice when computational resources are limited.

<!-- end list -->

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the model to be tuned
rf = RandomForestClassifier(random_state=42)

# Define the parameter distribution to sample from
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 30)
}

# Use RandomizedSearchCV to find the best hyperparameters
rscv = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10, # Number of random combinations to try
    scoring='roc_auc',
    cv=5,
    random_state=42
)

rscv.fit(X_train, y_train)
best_params = rscv.best_params_
```

---

## 3\. 🚀 High-Performance Modeling with LightGBM

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is known for its speed and efficiency, making it a popular choice for both simple and complex classification tasks.

- **Speed:** LightGBM builds decision trees "leaf-wise" rather than "level-wise," which often leads to faster training and better accuracy.
- **Performance:** It is highly effective with large datasets and often provides state-of-the-art results with minimal hyperparameter tuning.
- **Integration:** It integrates seamlessly into the `scikit-learn` ecosystem, allowing it to be used within pipelines and cross-validation routines.

<!-- end list -->

```python
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

# Create a LightGBM classifier with key parameters
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1, # Allows trees to grow to full depth
    random_state=42
)

# You can fit the model directly or within a pipeline
pipeline = Pipeline(steps=[('classifier', lgbm)])
pipeline.fit(X_train, y_train)
```

---

## 4\. 💾 Saving and Deploying a Complete Model Pipeline

Once a model is trained, it must be saved to a file to be used later for predictions without needing to be retrained. Saving the entire `Pipeline` object is a critical best practice.

- The `joblib` library is the recommended tool for saving `scikit-learn` objects. It is more efficient than the standard `pickle` module for objects containing large NumPy arrays.
- By saving the entire pipeline, you ensure that the same preprocessing steps used for training are automatically applied to new, raw data during prediction, guaranteeing consistency.

<!-- end list -->

```python
import joblib

# Assuming 'final_pipeline' is your fitted pipeline
# Save the entire pipeline to a file
joblib.dump(final_pipeline, 'model_pipeline.joblib')

# Later, in a new script or application, load the model
loaded_pipeline = joblib.load('model_pipeline.joblib')

# Use the loaded pipeline to make a prediction on new, raw data
new_data = pd.DataFrame(...)
prediction = loaded_pipeline.predict(new_data)
```
