---
title: markmap
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 4
  maxWidth: 350
---

# Domain 3: Modeling (36% of Exam)

## Task Statement 3.1: ML Problem Framing

### 1. The ML Pipeline Context
- **Previous Step (Domain 2)**: Data Preprocessing & Visualization
- **Current Step (This Course)**: Building, Training & Testing the ML Model
- **Exam Focus**: Choosing the right algorithm for a business use case and knowing its performance metrics.

### 2. When to Use and Not to Use ML

#### When ML Might Be a Good Fit
- **Requires Significant Data**: ML needs large datasets to build a predictable model.
- **Requires Expertise**: Needs data processing and feature engineering to handle noise and retain meaningful information.
- **Requires Powerful & Scalable Machines**: Needs computational power to crunch data and scale as data grows.

#### When ML is NOT the Right Solution
- **Mission-Critical Applications**: Scenarios where prediction errors are unacceptable.
- **Simple Problems**: Can be solved with simple rules and traditional programming (e.g., rule engines).
- **Be Cognizant**: ML is an expensive solution; ensure it's the right choice.

### 3. Identifying the Right Learning Type

#### Supervised Learning 👨‍🏫
- **Analogy**: A child learning under the guidance of a teacher.
- **Data**: Uses **labeled training data** (the outcome is already known).
- **Goal**: Model the relationship between inputs and outputs to predict new outcomes.
- **Feedback**: Possible, as the difference between actual and predicted values can be computed.

#### Unsupervised Learning 🕵️
- **Analogy**: A child figuring things out without supervision.
- **Data**: Deals with **unlabeled data**.
- **Goal**: Discover hidden structures, patterns, or information on its own.
- **Feedback**: Not possible.

#### Reinforcement Learning 🤖
- **Analogy**: Rewarding a kid for good behavior to reinforce it.
- **Core Idea**: An autonomous, self-learning agent learns through trial and error in an interactive environment to achieve the most optimal results.
- **Data**: No labeled data; learns from feedback (rewards/penalties).
- **Key Terminologies**:
    - **Agent**: The entity learning to make decisions (e.g., a player in a maze).
    - **Environment**: The problem space where the agent operates (e.g., the maze).
    - **State**: The agent's current condition/position.
    - **Action**: A choice made by the agent (e.g., move up, down, left, right).
    - **Reward/Penalty**: Feedback from the environment based on an action.
    - **Policy**: The agent's decision-making strategy to maximize rewards.
- **Types**:
    - **Model-Based**: Agent builds an internal model (a "mental map") of the environment to plan actions.
    - **Model-Free**: Agent learns directly through trial and error without building an explicit model.
- **Use Cases**: Self-driving vehicles, robotics, dynamic pricing, supply chain optimization.

### 4. Selecting the Right Model Type

#### Classification
- **Use Case**: When the dependent feature is categorical (discrete classes).
- **Analogy**: A fruit vendor sorting fruits as 'good' or 'bad'.
- **Types**:
    - **Binary Classification**: Two possible outcomes (e.g., Yes/No, Fraud/Legitimate).
        - *Algorithms*: Logistic Regression, Support Vector Machines (SVM).
    - **Multiclass Classification**: More than two mutually exclusive classes (e.g., sorting fruits into 'apples', 'oranges', 'bananas').
        - *Algorithms*: K-Nearest Neighbors (KNN), Naive Bayes.
    - **Multilabel Classification**: A data point can be assigned multiple labels (e.g., tagging a movie with genres like 'action', 'comedy', 'sci-fi').
        - *Algorithms*: Ensemble methods, Deep Learning.
- **Challenge: Imbalanced Data**
    - **Problem**: One class (minority) is significantly smaller than another (majority), causing biased models.
    - **Solution**: **SMOTE** (Synthetic Minority Over-sampling TEchnique) to generate new synthetic samples for the minority class.
- **Learner Types**:
    - **Eager Learners**: Build a model during training; fast predictions (e.g., Logistic Regression, SVM).
    - **Lazy Learners**: Memorize training data; slow predictions (e.g., KNN).
- **Common Algorithms**:
    - **Logistic Regression**: Predicts a binary outcome using a sigmoid (S-shaped) curve. Computationally efficient but sensitive to outliers.
    - **Naive Bayes**: Based on Bayes' theorem with an assumption of feature independence. Fast and handles missing data, but the independence assumption is a drawback.
    - **Support Vector Machines (SVM)**: Finds an optimal hyperplane to separate classes. Generalizes well but is computationally expensive and memory-intensive.
    - **K-Nearest Neighbors (KNN)**: Classifies new data based on the labels of its 'K' nearest neighbors. No training phase but is memory-intensive.

#### Regression
- **Use Case**: When the target feature is quantitative or continuous.
- **Example**: Predicting house prices based on features like square footage.
- **Variations**:
    - **Linear Regression**: One dependent and one independent feature (`Y = mx + b`).
    - **Multiple Regression**: One dependent and multiple independent features (`Y = m1x1 + m2x2 + ... + b`).
    - **Polynomial Regression**: Used when the relationship is non-linear (e.g., `Y = m1x1^2 + m2x2 + b`).

#### Time Series Forecasting 📈
- **Use Case**: For data collected at regular time intervals (e.g., stock prices, sales data).
- **Key Feature**: **Time** is always an independent feature.
- **Core Components**:
    - **Trend**: The direction of data over time (upward, downward).
    - **Seasonality**: Periodic fluctuations at regular intervals (e.g., higher sales in winter).
    - **Cyclical Variations**: Non-repeating fluctuations at irregular intervals (e.g., economic cycles).
    - **Irregularity**: Randomness or noise in the data.
- **Data Types**:
    - **Stationary Data**: Statistical properties (mean, variance) are constant over time. Easier to model.
    - **Non-Stationary Data**: Statistical properties change over time. Requires transformation (e.g., differencing) to make it stationary.
- **Limitations**: Sensitive to missing data, assumes linear relationships sometimes, relies heavily on historical data.

#### Clustering
- **Type**: Unsupervised learning algorithm.
- **Goal**: Group similar data points into clusters based on similarity (distance).
- **Hard vs. Soft Clustering**:
    - **Hard Clustering**: Each data point belongs to only one cluster (e.g., K-Means).
    - **Soft Clustering**: Each data point has a probability of belonging to each cluster (e.g., Fuzzy C-Means).
- **Categories**:
    - **Centroid-based (e.g., K-Means)**: Requires a predetermined number of clusters (`k`).
    - **Density-based (e.g., DBScan)**: Does not require a predetermined number of clusters; good at handling outliers.
    - **Hierarchical**: Builds a hierarchy of clusters (a `dendrogram`). Can be agglomerative (bottom-up) or divisive (top-down).
    - **Distribution-based (e.g., Gaussian Mixture Model)**: Assumes data comes from a mixture of probability distributions.
- **Use Cases**: Customer segmentation, anomaly/fraud detection, document organization.

#### Association Learning
- **Type**: Unsupervised learning.
- **Goal**: Discover hidden patterns and relationships between features (e.g., "if a customer buys X, they are likely to buy Y").
- **Analogy**: A vendor notices people who buy apples and oranges also buy bananas, so they place them together.
- **Terminology**:
    - **Rule**: `If {Antecedent} -> Then {Consequent}`
    - **Metrics**:
        - **Support**: Frequency of items occurring together.
        - **Confidence**: Likelihood that the consequent is purchased when the antecedent is.
        - **Lift**: Strength of the association (>1 means positive correlation).
- **Algorithms**: Apriori, FP-Growth, Eclat.
- **Use Cases**: Market basket analysis, recommendation systems, supply chain management.

#### Advanced: Deep Learning & Neural Networks 🧠
- **Purpose**: A subset of ML that addresses limitations of traditional ML, like handling large/complex data and non-linear relationships.
- **Concept**: Modeled after the human brain, using artificial neural networks with multiple layers.
- **Artificial Neuron (Perceptron)**:
    - **Components**: Inputs, Weights, Bias, Net Sum, Activation Function.
    - **Analogy to Biological Neuron**: Dendrites (Inputs), Axon (Output), Synapse (Weights).
- **Activation Functions**: Introduce non-linearity.
    - *Examples*: Sigmoid (S-shaped curve for probability), Tanh, **ReLU** (Rectified Linear Unit - fast and effective), Softmax.
- **Neural Network Architecture**:
    - **Layers**: Input Layer, Hidden Layer(s), Output Layer.
    - **Training Process**:
        - **Forward Propagation**: Data flows from input to output.
        - **Cost Function**: Measures the error in prediction.
        - **Backward Propagation**: Error is fed back to adjust weights and biases.
- **Challenges**: Requires high computational power, large amounts of data, and careful hyperparameter tuning.
- **Techniques**:
    - **CNN (Convolutional Neural Networks)**:
        - **Use Case**: Ideal for image and video processing.
        - **Architecture**:
            1. **Convolution Layer**: Applies filters (kernels) to extract features (creates a feature map).
            2. **Pooling Layer**: Downsamples the feature map (e.g., MaxPooling).
            3. **Flattening Layer**: Converts 2D data to a 1D vector.
            4. **Fully Connected Layer**: Performs classification based on extracted features.
    - **RNN (Recurrent Neural Networks)**:
        - **Use Case**: Designed to process sequential data (text, time series, audio) by using an internal memory.
        - **Key Idea**: Information flows in loops, allowing the network to persist information from previous steps.
        - **Architectures**: One-to-One, One-to-Many (Image Captioning), Many-to-One (Sentiment Analysis), Many-to-Many (Translation).
        - **Advanced RNNs**:
            - **LSTM (Long Short-Term Memory)**: Solves RNN's problem with long-range dependencies using a system of 'gates'.
            - **GRU (Gated Recurrent Unit)**: Similar to LSTM but with a simpler architecture, making it computationally less expensive.
    - **Transfer Learning**:
        - **Concept**: Reusing a pre-trained model on a new, related task.
        - **Analogy**: A Java expert learning Python reuses fundamental programming concepts.
        - **Process**: Freeze the weights of early layers (feature extraction) and retrain the final task-specific layers on new data.
        - **Benefits**: Reduces data scarcity issues, saves computational resources, and enhances performance.

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