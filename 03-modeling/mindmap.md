---
title: markmap
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 4
  maxWidth: 300
---

# Certification Exam Refresher

## Task Statement 3.1: ML Problem Framing

### 1. Determine When to Use and When Not to Use ML

#### When ML Might Be a Good Fit
- **Requires Significant Data**: ML needs large datasets to build a predictable model.
- **Requires Expertise**: Needs data processing and feature engineering to handle noise.
- **Requires Powerful Machines**: Needs computational power to crunch data.

#### When ML is NOT the Right Solution
- **Mission-Critical Applications**: Scenarios where prediction errors are unacceptable.
- **Simple Problems**: Can be solved with simple rules and traditional programming (e.g., rule engines).
- **Be Cognizant**: ML is an expensive solution; ensure it's the right choice.

### 2. Supervised vs. Unsupervised Learning

#### Supervised Learning
- **Data**: Uses labeled training data.
- **Features**: Dependent and independent features are clearly defined.
- **Primary Task**: Predict or classify new observations based on learned patterns.
- **Feedback**: Possible, as the difference between actual and predicted values can be computed.
- **Common Techniques**:
  - Linear Regression
  - Logistic Regression
  - Time Series Forecasting

#### Unsupervised Learning
- **Data**: Deals with unlabeled data.
- **Features**: Target feature is not available.
- **Primary Task**: Uncover hidden structures or patterns without explicit guidance.
- **Feedback**: Not possible.
- **Common Techniques**:
  - Clustering
  - Association Learning
  - Dimensionality Reduction

### 3. Select the Right Model Type

#### Classification
- **Use Case**: When the dependent feature is categorical.
- **Variations**:
  - **Binary Classification**: Target has two outcomes (e.g., yes/no, true/false).
  - **Multiclass Classification**
  - **Multilabel Classification**

#### Regression
- **Use Case**: When the target feature is quantitative or continuous.
- **Variations**:
  - Linear Regression
  - Multiple Regression
  - Polynomial Regression

#### Time Series Forecasting
- **Use Case**: For data collected at regular time intervals.
- **Core Components**:
  - **Trend**: Directionality of data over time.
  - **Seasonality**: Periodic fluctuations at regular intervals.
  - **Cyclical Variations**: Non-repeating fluctuations at irregular intervals.
  - **Irregularity**: Randomness or noise in the data.

#### Clustering
- **Type**: Unsupervised learning algorithm.
- **Goal**: Group similar data points into clusters based on similarity (distance).
- **Categories**:
  - **Centroid-based**: Requires a predetermined number of clusters.
  - **Density-based**: Does not require a predetermined number of clusters.
  - **Hierarchical**: Builds a hierarchy of clusters based on similarities.
  - **Distribution-based**: Assumes data is from a mixture of probability distributions.

#### Advanced Techniques: Deep Learning
- **Purpose**: Addresses limitations of traditional ML, like handling large/complex data and non-linear relationships.
- **Concept**: A subset of ML modeled after the human brain, using neural networks with multiple layers.
- **Techniques**:
  - **CNN (Convolutional Neural Networks)**: Uses layers to extract features for classification/detection.
  - **RNN (Recurrent Neural Networks)**: Designed to process sequential data using internal memory.
  - **Transfer Learning**: Leverages knowledge from a pre-trained model on one task to improve performance on a new, related task.

---

## Task Statement 3.2: Amazon SageMaker Built-in Algorithms

### Guiding Principle
- The data type in the question will help narrow down your algorithm choices.

### 1. Algorithms for Tabular Data
- **Definition**: Datasets organized in tables (rows/observations, columns/features).
- **Algorithms**:
  - XGBoost
  - Linear Learner
  - K-Nearest Neighbor (KNN)
  - Factorization Machines

### 2. Algorithms for Time Series Data
- **Definition**: Data recorded over consistent time intervals.
- **Examples**: Forecasting product demand, analyzing server loads.
- **Algorithm**:
  - DeepAR

### 3. Algorithms for Unsupervised Learning
- **Use Cases**: Unlabeled data for clustering, dimensionality reduction, anomaly detection.
- **Algorithms**:
  - Principal Component Analysis (PCA)
  - Random Cut Forest
  - IP Insights
  - K-Means

### 4. Algorithms for Text Data (NLP)
- **Use Cases**: Document summarization, topic modeling, language translation.
- **Algorithms**:
  - Object2Vec
  - Latent Dirichlet Allocation (LDA)
  - Neural Topic Model (NTM)
  - BlazingText
  - Sequence-to-Sequence

### 5. Algorithms for Image Data
- **Use Cases**: Analyze and process image data.
- **Algorithms**:
  - Image Classification
  - Object Detection
  - Semantic Segmentation

---

## Task Statement 3.3: Model Training & Optimization

### 1. Splitting Data
- **Training Data**: Used to train the model.
- **Validation Data (Optional)**: Measures model performance during training and tunes hyperparameters.
- **Testing Data**: Determines how well the model generalizes to unseen data.
- **Cross-Validation**: Process of validating the model against fresh, unseen data.
  - **Techniques**:
    - K-fold Cross-Validation
    - Stratified K-fold Cross-Validation
    - Leave-one-out Cross-Validation

### 2. Optimization Techniques
- **Loss Function**: Measures model accuracy (difference between predicted and actual output).
- **Optimization Goal**: Minimize the loss function for quick model convergence.
- **Gradient Descent**: A common optimization technique.
  - **Challenge**: May get stuck in local minima.
  - **Solutions**:
    - Stochastic Gradient Descent
    - Batch Gradient Descent

### 3. Choosing Compute Resources
- **CPUs are good for**:
  - Simpler classification/regression problems.
  - Smaller models with fewer parameters.
  - Low latency requirements.
  - Severe budget constraints.
- **GPUs are good for**:
  - Complex models (Deep Learning).
  - Models with a large number of parameters.
  - High throughput requirements.
  - Significant performance requirements.
- **Distributed Training**: For complex models on large datasets across multiple instances.
  - **SageMaker Strategies**: Data Parallelism & Model Parallelism.
  - **SageMaker Tools**: Prebuilt Docker images with Apache Spark.

### 4. Updating & Retraining Models
- **Importance**: Retraining with the latest data keeps models up-to-date and accurate.
- **Amazon SageMaker Canvas**:
  - Drag-and-drop UI for non-technical users.
  - Offers features for manual or automatically scheduled model updates.

---

## Task Statement 3.4: Hyperparameter Tuning & Model Concepts

### 1. Regularization
- **Goal**: Prevent overfitting and improve model performance.
- **Overfitting**: Performs well on training data, poorly on new data.
- **Underfitting**: Unable to learn hidden patterns in the data.
- **Techniques**:
  - **L1 Regularization**: Adds sum of *absolute* values of coefficients to the loss function. Good for minimizing impact of irrelevant features.
  - **L2 Regularization**: Adds sum of *squared* values of coefficients to the loss function. Distributes the impact of all important features.
  - **Early Stopping**: Stops training when performance on a validation set stops improving.

### 2. Cross-Validation
- **Goal**: Prevent overfitting by assessing performance and generalizability, especially with limited data.
- **Concept**: Partitions the dataset into training and testing subsets to train on different parts of the data.
- **Techniques**:
  - **K-fold**: Dataset split into K folds. Train on K-1, test on the Kth fold. Repeat K times.
  - **Stratified K-fold**: Like K-fold, but each fold maintains the same class distribution as the original dataset. Effective for imbalanced data.
  - **Time Series**: Folds are created sequentially based on time, ensuring chronological order. Effective when data order is important.

### 3. Initializing Models for Tuning
- **Process**: Initialize algorithms with a starting set of hyperparameter values.
- **Ranges**: Tuning jobs search for the best values over defined ranges.
- **Range Types**:
  - Categorical Parameter
  - Continuous Parameter
  - Integer Parameter

### 4. Neural Network Architecture
- **Components**:
  - **Input Layer**: Receives data.
  - **Hidden Layer(s)**: Transform input data.
  - **Output Layer**: Makes final predictions.
- **Key Concepts**:
  - **Weights**: Determine feature importance; adjusted during training.
  - **Biases**: One per neuron; helps the model fit the data better.
  - **Activation Function**: Introduces non-linearity to learn complex patterns.
    - *Examples*: Sigmoid, Tanh, ReLU, Softmax.

### 5. Understanding Tree-Based Models
- **Type**: Supervised learning algorithm that builds a tree structure.
- **Structure**:
  - **Root Node**: Top node, represents the entire dataset.
  - **Sub-nodes**: Intermediate decision points that split the data.
  - **Branches/Edges**: Represent the outcomes of a decision.
  - **Leaf Nodes**: Terminal nodes representing the final outcome.
- **Key Hyperparameters**:
  - `no_of_folds`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`

### 6. Understanding Linear Models
- **Gradient Descent**: Advanced algorithm to find optimal hyperparameter values.
- **Key Hyperparameters**:
  - **Learning Rate**: Critical for the efficiency of the optimization process.
  - **Alpha**: Controls regularization strength in Ridge Regression to balance bias and variance.

---

## Task Statement 3.5: Model Evaluation

### Note on Overlapping Topics
- Overfitting, underfitting, and cross-validation are covered in detail under **Task Statement 3.4**.

### 1. Evaluate Metrics
- **Classification Metrics (When to Use)**:
  - **Accuracy**: For balanced datasets where error costs are similar.
  - **Precision**: For imbalanced datasets with a high cost of false positives (e.g., fraud/spam detection).
  - **Recall**: For imbalanced datasets with a high cost of false negatives (e.g., disease detection).
  - **F1 Score**: To balance both precision and recall (e.g., text classification).
  - **AUC Curve**: For binary classification to evaluate a model's discrimination ability.
- **Regression Metrics (When to Use)**:
  - **Mean Absolute Error (MAE)**: When data has outliers and you don't want them to have a disproportionate effect.
  - **Mean Squared Error (MSE)**: When large errors are more problematic than small ones.
  - **RMSE**: Like MSE but in the same units as the target variable; used when large errors are significantly more problematic.
  - **MAPE**: To express errors as a percentage.

### 2. Interpret Confusion Matrices
- **Purpose**: Not a metric itself, but forms the basis for many other performance metrics.
- **Components**:
  - **True Positive (TP)**: Model correctly predicted the positive class.
  - **True Negative (TN)**: Model correctly predicted the negative class.
  - **False Positive (FP) / Type 1 Error**: Model incorrectly predicted the positive class.
  - **False Negative (FN) / Type 2 Error**: Model incorrectly predicted the negative class.

### 3. Perform Online and Offline Model Evaluation
- **Online Evaluation**:
  - **Concept**: Continuously assessing a model's performance using live data in production.
  - **Performance Metrics**: Latency, Throughput, Data Drift.
  - **Business Metrics**: Clickthrough Rate, Conversion Rate.
  - **A/B Testing**: Evaluating multiple model versions by running them in parallel and distributing traffic.

### 4. Compare ML Models
- **Beyond standard metrics**, consider computational complexity.
- **Computational Complexity Metrics**:
  - **Time Complexity**: Time taken by an algorithm for a given input size.
  - **Space Complexity**: Amount of additional memory an algorithm needs.
  - **Sample Complexity**: Number of training samples needed to achieve desired performance.
  - **Parametricity**: Whether a model has a fixed or dynamic number of parameters.