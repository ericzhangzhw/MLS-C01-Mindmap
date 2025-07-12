---
title: markmap
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
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

#### Unsupervised Learning 🕵️
- **Analogy**: A child figuring things out without supervision.
- **Data**: Deals with **unlabeled data**.
- **Goal**: Discover hidden structures, patterns, or information on its own.

#### Reinforcement Learning 🤖
- **Analogy**: Rewarding a kid for good behavior to reinforce it.
- **Core Idea**: An autonomous agent learns through trial and error in an interactive environment.
- **Use Cases**: Self-driving vehicles, robotics, dynamic pricing.

### 4. Selecting the Right Model Type

#### Classification
- **Use Case**: When the dependent feature is categorical (discrete classes).
- **Types**: Binary, Multiclass, Multilabel.
- **Challenge**: Imbalanced Data (Solved with **SMOTE**).
- **Common Algorithms**: Logistic Regression, Naive Bayes, SVM, KNN.

#### Regression
- **Use Case**: When the target feature is quantitative or continuous.
- **Example**: Predicting house prices.
- **Variations**: Linear, Multiple, Polynomial Regression.

#### Time Series Forecasting 📈
- **Use Case**: For data collected at regular time intervals.
- **Key Feature**: **Time** is always an independent feature.
- **Core Components**: Trend, Seasonality, Cyclical Variations, Irregularity.

#### Clustering
- **Type**: Unsupervised learning.
- **Goal**: Group similar data points into clusters.
- **Categories**: Centroid-based (K-Means), Density-based (DBScan), Hierarchical.

#### Association Learning
- **Type**: Unsupervised learning.
- **Goal**: Discover "if-then" relationships between features.
- **Use Cases**: Market basket analysis, recommendation systems.

#### Advanced: Deep Learning & Neural Networks 🧠
- **Purpose**: A subset of ML for large/complex data and non-linear relationships.
- **CNN (Convolutional Neural Networks)**: For image/video processing.
- **RNN (Recurrent Neural Networks)**: For sequential data (text, time series). Includes LSTM & GRU.
- **Transfer Learning**: Reusing a pre-trained model on a new, related task to save resources and improve performance.

---

## Task Statement 3.2: Amazon SageMaker Built-in Algorithms

### 1. Introduction to the SageMaker Ecosystem
- **Overview**: A fully managed service to prepare, build, train, and deploy ML models at scale.
- **Data Collection**: **SageMaker Ground Truth** for data labeling.
- **Data Analysis/Prep**:
  - **SageMaker Data Wrangler**: Visualize and prepare data with no code.
  - **SageMaker Feature Store**: Store and retrieve features for model development.
- **Model Building**:
  - **SageMaker Notebooks**: Managed Jupyter Notebooks.
  - **SageMaker Studio**: An IDE for the entire ML lifecycle.
- **Model Training**:
  - **Options**: Use built-in algorithms, script mode (scikit-learn, PyTorch), or a custom Docker image.
  - **Architecture**:
    - Training data stored in **S3**.
    - Training job runs on **SageMaker compute instances**.
    - Training job stored in **Amazon ECR**.
    - Model output stored in another **S3 bucket**.
    - *Constraint*: Training data and job must be in the same AWS region.
- **Model Deployment**:
  - **SageMaker Hosting Services**: For real-time inference with low latency.
  - **SageMaker Batch Transform**: For asynchronous batch inference.
- **Model Monitoring**:
  - **SageMaker Model Monitor**: Continuously monitors deployed models for performance and drift.

### 2. Algorithms for Tabular Data
- **XGBoost (Extreme Gradient Boosting)**
  - **Concept**: A popular and efficient implementation of gradient-boosted decision trees.
    - **Ensemble Learning**: Combines multiple "weak" models to create one strong model.
    - **Boosting**: A sequential technique where models are trained to correct the errors of their predecessors.
    - **Gradient Boosting**: Uses gradient descent to minimize errors.
  - **Problem Type**: Classification & Regression.
  - **Data Formats**: `libsvm`, `CSV`, `parquet`, `protobuf`.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_round`, `num_class`.
  - **Metrics**: MAE, MSE, RMSE (Regression); Accuracy, AUC, F1 Score (Classification).
  - **Use Cases**: Fraud detection, stock price prediction, customer churn, sales forecasting.
- **Linear Learner**
  - **Concept**: A supervised algorithm for classification or regression, great for large, high-dimensional datasets. It fits a line to the data points by adjusting weights and biases using stochastic gradient descent.
  - **Problem Type**: Classification & Regression.
  - **Data Formats**: `protobuf`, `CSV` for training. `JSON`, `protobuf`, `CSV` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_classes`, `predictor_type`.
  - **Metrics**: Cross-entropy loss, MAE, MSE (Regression); Precision, Recall, Accuracy (Classification).
  - **Use Cases**: Loan application processing, email spam detection, recommendation systems.
- **K-Nearest Neighbor (KNN)**
  - **Concept**: A non-parametric algorithm that classifies a data point based on its "K" closest neighbors. For regression, it averages their values; for classification, it uses a majority vote.
  - **Problem Type**: Classification & Regression.
  - **Data Formats**: `protobuf`, `CSV` for training. `JSON`, `protobuf`, `CSV` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `feature_dim`, `k`, `predictor_type`, `sample_size`.
  - **Metrics**: MSE (Regression); Accuracy (Classification).
  - **Use Cases**: Credit risk rating, recommendation systems, fraud detection.
- **Factorization Machines**
  - **Concept**: An extension of a linear model designed to capture higher-order (pairwise) feature interactions in sparse datasets.
  - **Problem Type**: Binary Classification & Regression.
  - **Limitations**: Only considers pairwise features, does not support multiclass problems or CSV format, performs poorly on dense data.
  - **Data Formats**: `protobuf` (float32 tensors) only for training. `JSON`, `protobuf` for inference.
  - **Compute**: Recommended for CPU only.
  - **Required Hyperparameters**: `feature_dim`, `num_factors`, `predictor_type`.
  - **Metrics**: RMSE (Regression); Accuracy, Cross-entropy (Classification).
  - **Use Cases**: Recommendation systems, ad-click prediction.

### 3. Algorithms for Time Series Data
- **DeepAR**
  - **Concept**: A supervised algorithm using RNNs for forecasting one-dimensional time series data. Can learn from multiple related time series to solve the "cold start problem" for new products.
  - **Forecast Types**: Point-in-time (single value) and Probabilistic (range of values).
  - **Data Format**: `JSON Lines` format (can be GZIP or Parquet). Requires `start` and `target` fields.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `context_length`, `epochs`, `prediction_length`, `time_freq`.
  - **Metrics**: RMSE, `wQuantileLoss`.
  - **Use Cases**: Demand/sales forecasting, financial forecasting, risk assessment.

### 4. Algorithms for Unsupervised Learning
- **Principal Component Analysis (PCA)**
  - **Concept**: Reduces the number of features (dimensionality) in a dataset by creating new, uncorrelated features called "components" that capture the most variance.
  - **Problem Type**: Dimensionality Reduction.
  - **Modes**: `regular` (for sparse data) and `randomized` (for large datasets).
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `feature_dim`, `mini_batch_size`, `num_components`.
  - **Use Cases**: Image compression, financial analysis, customer feedback analysis.
- **Random Cut Forest (RCF)**
  - **Concept**: Detects anomalies (outliers) by building an ensemble of trees. Data points that are easily isolated with fewer "cuts" are assigned a higher anomaly score.
  - **Problem Type**: Anomaly Detection.
  - **Data Formats**: `protobuf`, `CSV`.
  - **Compute**: Recommended for CPU only.
  - **Required Hyperparameters**: `feature_dim`.
  - **Metrics**: F1 Score.
  - **Use Cases**: Fraud detection, security breach detection, monitoring bot activity.
- **IP Insights**
  - **Concept**: Learns usage patterns for IPv4 addresses by associating them with entities (e.g., user IDs). Uses a neural network to detect anomalous logins from unusual IP addresses or locations.
  - **Problem Type**: Anomaly Detection.
  - **Data Formats**: `CSV` for training. `CSV`, `JSON`, `JSON Lines` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_entity_vectors`, `vector_dim`.
  - **Metrics**: Area Under Curve (AUC).
  - **Use Cases**: Detecting fraudulent transactions/account takeovers, compliance checks, geolocation-based personalization.
- **K-Means**
  - **Concept**: Groups data into a pre-determined number (`K`) of clusters. It iteratively assigns data points to the nearest cluster centroid and then recalculates the centroid.
  - **Problem Type**: Clustering.
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON` for inference.
  - **Compute**: Recommends CPU (GPU supported for single instance only).
  - **Required Hyperparameters**: `feature_dim`, `K`.
  - **Metrics**: Mean Square Distance (msd), Sum of Square Distance (ssd).
  - **Use Cases**: Customer segmentation, market segmentation, recommendation systems.

### 5. Algorithms for Text Data (NLP)
- **Object2Vec**
  - **Concept**: A customizable neural embedding algorithm that creates vector representations of objects (e.g., words, sentences, users, products) by learning from their relationships.
  - **Problem Type**: General Purpose Embedding.
  - **Data Formats**: `JSON Lines` (sentence-sentence or label-sentence pairs) for training. `JSON` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `enc0_max_seq_len`, `enc0_vocab_size`.
  - **Metrics**: MSE (Regression); Accuracy, Cross-entropy (Classification).
  - **Use Cases**: User behavior analysis, sentiment analysis, social network analysis.
- **Latent Dirichlet Allocation (LDA)**
  - **Concept**: An unsupervised generative probabilistic model that discovers underlying topics in a collection of documents.
  - **Problem Type**: Topic Modeling.
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON` for inference.
  - **Compute**: Supports single-instance CPU only.
  - **Required Hyperparameters**: `num_topics`, `feature_dim`, `mini_batch_size`.
  - **Metrics**: Per-Word-Log-Likelihood (pwll).
  - **Use Cases**: Customer feedback analysis, social media trend analysis, content creation ideas.
- **Neural Topic Model (NTM)**
  - **Concept**: An unsupervised algorithm, similar to LDA, but uses a neural network to model topics. More scalable than LDA but can be less interpretable.
  - **Problem Type**: Topic Modeling.
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON`, `JSON Lines` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_topics`, `feature_dim`.
  - **Metrics**: Total loss.
  - **Use Cases**: Uncovering customer pain points, personalized content recommendations, market sentiment analysis.
- **BlazingText**
  - **Concept**: A highly optimized implementation of `Word2Vec` (unsupervised) and `FastText` (supervised text classification). Significantly faster than original implementations.
  - **Modes**: `word2vec` (cbow, skip-gram) and `text classification`.
  - **Data Formats**: A single preprocessed text file (space-separated tokens). `JSON` for inference.
  - **Compute**: Supports single CPU/GPU; multiple CPUs for batch_skip-gram.
  - **Required Hyperparameters**: `mode`.
  - **Metrics**: `mean_rho` (Word2Vec); Accuracy (Text Classification).
  - **Use Cases**: Sentiment analysis, document classification, recommendation systems.
- **Sequence-to-Sequence (Seq2Seq)**
  - **Concept**: A supervised algorithm that transforms an input sequence to an output sequence using an encoder-decoder neural network architecture.
  - **Problem Type**: Language Processing (Translation, Summarization).
  - **Data Formats**: `protobuf` for training. `JSON`, `protobuf` for inference.
  - **Compute**: Supports single-machine GPU only.
  - **Required Hyperparameters**: None.
  - **Metrics**: Accuracy, BLEU score, Perplexity.
  - **Use Cases**: Machine translation, speech-to-text conversion, code generation.

### 6. Algorithms for Image Data
- **Image Classification**
  - **Concept**: A supervised algorithm that classifies an entire image into one or more categories. Uses a Convolutional Neural Network (CNN).
  - **Modes**: `Full training` (from scratch) or `Transfer learning` (using pre-trained weights).
  - **Data Formats**: `recordIO` or image formats (`JPG`, `PNG`), plus a `.lst` file listing images.
  - **Compute**: Recommends GPU for training.
  - **Required Hyperparameters**: `num_classes`, `num_training_samples`.
  - **Metrics**: Accuracy.
  - **Use Cases**: Medical image diagnosis (X-rays), classifying objects for autonomous vehicles, security surveillance.
- **Object Detection**
  - **Concept**: Goes beyond classification to identify and locate *multiple* objects within a single image by drawing bounding boxes around them. Uses Single Shot MultiBox Detector (SSD) framework.
  - **Data Formats**: `recordIO` or image formats (`JPG`, `PNG`), with a matching `.json` file for each image's annotations.
  - **Compute**: Recommends GPU for training.
  - **Required Hyperparameters**: `num_classes`, `num_training_samples`.
  - **Metrics**: Mean Average Precision (mAP).
  - **Use Cases**: Automated retail checkout systems, manufacturing quality control, scanning information from documents.
- **Semantic Segmentation**
  - **Concept**: The most granular image task, assigning a class label to *every single pixel* in an image to understand object shapes. The output is a segmentation mask.
  - **Data Formats**: Requires separate channels for `train`, `validation` images (`JPG`) and their corresponding `train_annotation`, `validation_annotation` label images (`PNG`).
  - **Compute**: Recommends GPU for training.
  - **Required Hyperparameters**: `num_classes`, `num_training_samples`.
  - **Metrics**: Mean Intersection-over-Union (mIOU), Pixel Accuracy.
  - **Use Cases**: Analyzing satellite imagery, retail shelf analysis, content moderation in media.

---

## Task Statement 3.3: Model Training & Optimization

### 1. Splitting Data
- **Training Data**: Used to train the model.
- **Validation Data**: Measures performance during training to tune hyperparameters.
- **Testing Data**: Determines how well the model generalizes to unseen data.
- **Cross-Validation**: Validating the model against fresh data using techniques like K-fold or Stratified K-fold.

### 2. Optimization Techniques
- **Loss Function**: Measures the difference between predicted and actual output.
- **Gradient Descent**: A common technique to minimize the loss function.

### 3. Choosing Compute Resources
- **CPUs**: Good for simpler models, small datasets, and budget constraints.
- **GPUs**: Good for complex deep learning models, large datasets, and high performance needs.
- **Distributed Training**: For very large models/datasets (SageMaker offers Data & Model Parallelism).

### 4. Updating & Retraining Models
- **Importance**: Retraining with new data keeps models accurate.
- **Amazon SageMaker Canvas**: Offers a no-code UI for scheduling model updates.

---

## Task Statement 3.4: Hyperparameter Tuning & Model Concepts

### 1. Regularization
- **Goal**: Prevent overfitting.
- **Techniques**: L1, L2, Early Stopping.

### 2. Cross-Validation
- **Goal**: Prevent overfitting by assessing performance on different data subsets.
- **Techniques**: K-fold, Stratified K-fold, Time Series split.

### 3. Initializing Models for Tuning
- **Process**: Define ranges (Categorical, Continuous, Integer) for hyperparameters for tuning jobs to search over.

### 4. Neural Network Architecture
- **Components**: Input Layer, Hidden Layer(s), Output Layer.
- **Key Concepts**: Weights, Biases, Activation Functions (ReLU, Softmax).

### 5. Understanding Tree-Based Models
- **Structure**: Root Node, Sub-nodes, Branches, Leaf Nodes.
- **Key Hyperparameters**: `max_depth`, `min_samples_split`.

### 6. Understanding Linear Models
- **Key Hyperparameters**: Learning Rate, Alpha (regularization strength).

---

## Task Statement 3.5: Model Evaluation

### 1. Evaluate Metrics
- **Classification**: Accuracy, Precision, Recall, F1 Score, AUC Curve.
- **Regression**: MAE, MSE, RMSE.

### 2. Interpret Confusion Matrices
- **Purpose**: A table showing the performance of a classification model.
- **Components**: True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN).

### 3. Online and Offline Model Evaluation
- **Online Evaluation (A/B Testing)**: Assessing model performance on live data in production.

### 4. Compare ML Models
- **Consider**: Computational Complexity (Time, Space), Sample Complexity.