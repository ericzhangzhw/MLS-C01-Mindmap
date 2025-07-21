---
title: Developer Guide
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 350
---

# Amazon Machine Learning: Developer Guide

## Task 1: Introduction & Concepts

### 1. The Amazon ML Service
- **What is Amazon ML?**
  - A cloud-based service to build, train, and host scalable ML models.
- **Current Status**
  - No longer accepting new users.
  - Existing users can continue to use the service and documentation.
  - **Successor**: **Amazon SageMaker AI** is the recommended robust, fully managed service for all skill levels.

### 2. Core Concepts of Amazon ML
- **Datasources**
  - An object containing metadata about input data (e.g., statistics, schema).
  - **Does not store** a copy of the data; it references an Amazon S3 location.
  - Used for training, evaluation, and batch predictions.
- **ML Models**
  - A mathematical model that generates predictions by finding patterns in data.
  - **Types Supported**
    - **Binary Classification**: Predicts one of two outcomes (e.g., true/false).
    - **Multiclass Classification**: Predicts from a limited, pre-defined set of values.
    - **Regression**: Predicts a numeric value.
- **Evaluations**
  - Measures the quality and performance of an ML model.
  - **Key Metrics**
    - **AUC (Area Under Curve)**: For binary model performance.
    - **Macro-averaged F1-score**: For multiclass model performance.
    - **RMSE (Root Mean Square Error)**: For regression model performance.
- **Predictions**
  - **Batch Predictions**: Asynchronously generate predictions for many observations at once.
  - **Real-time Predictions**: Synchronously generate low-latency predictions for individual observations.

### 3. Access & Pricing
- **Accessing Amazon ML**
  - **Amazon ML console**: A web-based user interface.
  - **AWS CLI**: Command line interface for managing services.
  - **Amazon ML API**: For programmatic access.
- **Pricing**
  - No minimum fees or upfront commitments.
  - **Data Analysis & Model Training**: Charged an hourly rate for compute time.
  - **Batch Predictions**: Charged per 1,000 predictions (e.g., $0.10/1000).
  - **Real-time Predictions**: Charged per prediction (e.g., $0.0001/prediction) plus an hourly reserved capacity charge based on model size.

---

## Task 2: The Machine Learning Process

### 1. Framing the ML Problem
- **Supervised Learning**
  - Learning from data that has been labeled with the correct answer.
  - Amazon ML focuses on supervised learning tasks.
- **Problem Types & Examples**
  - **Binary Classification**: "Is this email spam or not spam?"
  - **Multiclass Classification**: "Is this product a book, movie, or clothing?"
  - **Regression**: "What will the temperature be in Seattle tomorrow?"

### 2. Building an ML Application: Step-by-Step
- **Step 1: Formulate the Problem**
  - Decide what you want to predict (the target).
  - Frame it as the simplest solution that meets your needs (e.g., regression vs. binary classification).
- **Step 2: Collect Labeled Data**
  - Gather examples where you already know the target answer.
  - The data must contain both the **target** and the **variables/features** used to predict it.
- **Step 3: Analyze & Prepare Data**
  - **Analyze**: Inspect data summaries and visualizations to check quality and understand distributions.
  - **Format**: Convert data into a format acceptable to the algorithm, such as **comma-separated values (CSV)**.
- **Step 4: Feature Processing / Engineering**
  - Transform raw variables to make them more predictive.
  - **Examples**:
    - Binning numeric features into categories.
    - Creating Cartesian products of variables.
    - Forming n-grams from text features.
  - **Implementation**: In Amazon ML, this is done via a **recipe**.
- **Step 5: Split Data**
  - Split labeled data into **training** and **evaluation** subsets.
  - A common split is 70% for training and 30% for evaluation.
- **Step 6: Train the Model**
  - The learning algorithm (SGD) finds patterns in the training data and outputs a model.
  - **Hyperparameters (Training Parameters)**: Control the quality of the resulting model.
    - **Number of Passes**: How many times the algorithm iterates over the training data.
    - **Data Shuffling**: Must shuffle training data because SGD is sensitive to data order.
    - **Regularization**: Prevents overfitting. **L1** regularization creates sparse models (feature selection); **L2** stabilizes weights.
- **Step 7: Evaluate Model Accuracy**
  - Use the held-out evaluation data to assess model performance.
  - **Binary Classification**: Use the **AUC** metric. Adjust the **score threshold (cut-off)** to manage the trade-off between false positives and false negatives.
  - **Multiclass Classification**: Use the **macro average F1-score**. Review the **confusion matrix**.
  - **Regression**: Use the **RMSE** metric. Review the **distribution of residuals**.
- **Step 8: Use the Model to Make Predictions**
  - **Batch Predictions**: Generate predictions for a set of observations all at once.
  - **Real-time Predictions**: Generate low-latency predictions for individual observations.

---

## Task 3: Datasources, Training & Evaluation In-Depth

### 1. Datasources & Data Formatting
- **Data Format**
  - Input data must be in **.csv format**.
  - Each row is an observation; each column is an attribute.
  - A header line with attribute names is recommended.
- **Data Schema**
  - A schema allows Amazon ML to interpret the data correctly.
  - Can be inferred by Amazon ML or provided by the user.
  - **Key Attributes**:
    - `targetAttributeName`: The column to be predicted.
    - `rowId`: An optional unique identifier excluded from training but included in output.
    - `attributeType`: Defines variable type (BINARY, CATEGORICAL, NUMERIC, TEXT).
- **Data from AWS Services**
  - **Amazon Redshift**: Create a datasource from a SQL query. The service uses the **UNLOAD** command to copy data to S3.
  - **Amazon RDS**: Create a datasource from a MySQL database. The service creates an **AWS Data Pipeline** object to run the query and export data to S3.

### 2. Feature Transformations with Recipes
- **Purpose**: A recipe contains instructions for transforming features to optimize the data for learning.
- **Recipe Format**
  - **`groups`**: Define collections of variables to apply transformations collectively.
  - **`assignments`**: Define intermediate transformed variables.
  - **`outputs`**: Specifies all variables (raw and transformed) to be used by the learning process.
- **Key Transformations**
  - **`ngram(var, size)`**: Creates word combinations from text.
  - **`quantile_bin(var, n)`**: Bins a numeric feature into `n` categories.
  - **`cartesian(var1, var2)`**: Creates permutations of variables to capture interaction effects.
  - **`lowercase(var)`** and **`no_punct(var)`**: For text cleaning.

### 3. Training & Evaluation
- **Overfitting vs. Underfitting**
  - **Overfitting**: The model performs well on training data but poorly on new data because it has "memorized" the training set, including noise.
  - **Underfitting**: The model is too simple and performs poorly on both training and evaluation data because it can't capture the underlying patterns.
- **Preventing Overfitting**
  - **Regularization**: Penalizes complex models.
  - **Cross-Validation**: A robust method to assess performance by training and evaluating a model on multiple, non-overlapping data splits (k-folds) and averaging the results.
- **Evaluation Alerts**
  - Amazon ML provides alerts if the evaluation is not valid.
  - **Checks**:
    - Evaluation is done on held-out data (not the training data).
    - Sufficient data was used for evaluation.
    - Target distribution is similar between training and evaluation datasources.