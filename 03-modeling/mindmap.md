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
- **ML Lifecycle Services**:
  - **Data Collection**: **SageMaker Ground Truth** for building highly accurate, labeled training sets.
  - **Data Analysis/Prep**:
    - **SageMaker Data Wrangler**: Visualize and prepare data with no code.
    - **SageMaker Feature Store**: Simplifies feature processing, storing, and retrieving for model development.
  - **Model Building**:
    - **SageMaker Notebooks**: Managed Jupyter Notebooks.
    - **SageMaker Studio**: An IDE for the entire ML lifecycle.
  - **Model Training**:
    - **Architecture**:
        - Training data is stored in an **S3 bucket**.
        - Training job runs on **SageMaker compute instances**.
        - The training job (algorithm) is stored in **Amazon ECR**.
        - Model output (artifacts) is stored in another **S3 bucket**.
        - *Constraint*: Training data and job must be in the same AWS region.
    - **Implementation Options**:
        - **Built-in Algorithms**: Easiest option, requires no custom code.
        - **Script Mode**: Use a custom Python script with supported frameworks (scikit-learn, TensorFlow, PyTorch).
        - **Custom Docker Image**: For use cases not covered by the other options; requires Docker knowledge.
  - **Model Deployment**:
    - **SageMaker Hosting Services**: For real-time inference with low latency.
    - **SageMaker Batch Transform**: For asynchronous batch inference on large datasets.
  - **Model Monitoring**:
    - **SageMaker Model Monitor**: Continuously monitors deployed models for performance and concept drift.

### 2. Algorithms for Tabular Data
- **XGBoost (Extreme Gradient Boosting)**
  - **Concept**: A popular and efficient implementation of gradient-boosted decision trees.
    - **Ensemble Learning**: A "wisdom of the crowd" approach where multiple "weak" models are combined to create one strong model.
    - **Boosting**: A sequential ensemble technique. Models are trained in sequence, with each new model focusing on correcting the errors of its predecessor. Models with larger errors are given higher weights.
    - **Gradient Boosting**: A specific type of boosting that uses the gradient descent algorithm to minimize errors.
    - **Decision Tree**: Predicts an outcome by evaluating a sequence of "if-then-else" questions on features, creating branches until a final decision (leaf) is reached.
  - **Problem Type**: Classification & Regression.
  - **Data Formats**: `libsvm`, `CSV`, `parquet`, `protobuf`.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_round`, `num_class`.
  - **Metrics**: MAE, MSE, RMSE (Regression); Accuracy, AUC, F1 Score (Classification).
  - **Use Cases**: Fraud detection, stock price prediction, customer churn, sales forecasting, ad-click revenue.
- **Linear Learner**
  - **Concept**: A supervised algorithm for classification or regression, great for large, high-dimensional datasets. It fits a line to data points by adjusting weights (m) and biases (b) for each feature (e.g., Y = m1x1 + m2x2 + ... + b). It uses **stochastic gradient descent** to iteratively adjust these parameters and minimize the difference between predicted and actual values.
  - **Problem Type**: Classification & Regression.
  - **Data Formats**: `protobuf`, `CSV` for training. `JSON`, `protobuf`, `CSV` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_classes`, `predictor_type`.
  - **Metrics**: Cross-entropy loss, MAE, MSE (Regression); Precision, Recall, Accuracy (Classification).
  - **Use Cases**: Loan application processing, email spam detection, recommendation systems.
- **K-Nearest Neighbor (KNN)**
  - **Concept**: A non-parametric, index-based algorithm. It classifies a new data point based on the properties of its "K" closest neighbors.
    - **For Regression**: It averages the values of the K nearest neighbors (e.g., predicting a house price based on the prices of nearby, similar houses).
    - **For Classification**: It uses a majority vote among the K nearest neighbors (e.g., predicting a house has 3 bedrooms because most houses in its community do).
  - **Problem Type**: Classification & Regression.
  - **Data Formats**: `protobuf`, `CSV` for training. `JSON`, `protobuf`, `CSV` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `feature_dim`, `k`, `predictor_type`, `sample_size`.
  - **Metrics**: MSE (Regression); Accuracy (Classification).
  - **Use Cases**: Credit risk rating, recommendation systems, fraud detection.
- **Factorization Machines**
  - **Concept**: An extension of a linear model designed to capture higher-order (pairwise) feature interactions in sparse datasets, like recommendation systems. It uses latent vectors (numerical representations of hidden features like genre or actors) to model these interactions.
  - **Problem Type**: Binary Classification & Regression.
  - **Limitations**: Only considers pairwise features, does not support multiclass problems, does not support CSV format, performs poorly on dense data, and needs a lot of data (e.g., 10k-10M rows) to work around missing features.
  - **Data Formats**: `protobuf` (float32 tensors) only for training. `JSON`, `protobuf` for inference.
  - **Compute**: Recommended for CPU only.
  - **Required Hyperparameters**: `feature_dim`, `num_factors`, `predictor_type`.
  - **Metrics**: RMSE (Regression); Accuracy, Cross-entropy (Classification).
  - **Use Cases**: Recommendation systems, ad-click prediction.

### 3. Algorithms for Time Series Data
- **DeepAR**
  - **Concept**: A supervised algorithm using RNNs for forecasting one-dimensional time series data. It can learn from multiple related time series, which helps solve the "cold start problem" for new items that have no historical data.
  - **Forecast Types**:
      - **Point-in-time**: Predicts a single value (e.g., we will sell 1000 units).
      - **Probabilistic**: Predicts a range of values with a probability (e.g., we will sell 800-1200 units with 90% probability).
  - **Data Format**: `JSON Lines` format (can be GZIP or Parquet). Requires `start` (timestamp string) and `target` (array of values) fields. Optional fields include `dynamic_feat` (e.g., a boolean for a promotion) and `cat` (categorical features).
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `context_length`, `epochs`, `prediction_length`, `time_freq`.
  - **Metrics**: RMSE, `wQuantileLoss`.
  - **Use Cases**: Demand/sales forecasting, financial forecasting, risk assessment.

### 4. Algorithms for Unsupervised Learning
- **Principal Component Analysis (PCA)**
  - **Concept**: Reduces the number of features (dimensionality) in a dataset by creating new, uncorrelated features called "components" that capture the most variance, without losing meaningful information.
  - **Analogy**: Like taking photos of a 3D object from the best possible angles to capture all its important features in a 2D representation. These "best angles" are the principal components.
  - **Problem Type**: Dimensionality Reduction.
  - **Modes**: `regular` (for sparse data) and `randomized` (for large datasets).
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `feature_dim`, `mini_batch_size`, `num_components`.
  - **Use Cases**: Image compression, financial analysis, customer feedback analysis.
- **Random Cut Forest (RCF)**
  - **Concept**: Detects anomalies (outliers) by building an ensemble of trees. Data points that are easily isolated with fewer "cuts" are assigned a higher anomaly score.
  - **Analogy**: A forester takes random sample plots (random cuts) in a forest (dataset). A tree (data point) that stands alone after a cut is considered isolated and thus an outlier.
  - **Problem Type**: Anomaly Detection.
  - **Data Formats**: `protobuf`, `CSV`.
  - **Compute**: Recommended for CPU only.
  - **Required Hyperparameters**: `feature_dim`.
  - **Metrics**: F1 Score.
  - **Use Cases**: Detecting fraud in financial transactions, identifying security breaches in network traffic, monitoring for bot activity in e-commerce.
- **IP Insights**
  - **Concept**: Learns usage patterns for IPv4 addresses by associating them with entities (e.g., user IDs). Uses a neural network to detect anomalous logins from unusual IP addresses or locations and returns a high score for deviations.
  - **Example**: If you always log in from home and then suddenly log in from another country, the algorithm flags it as anomalous and can trigger additional security checks.
  - **Problem Type**: Anomaly Detection.
  - **Data Formats**: `CSV` for training. `CSV`, `JSON`, `JSON Lines` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_entity_vectors`, `vector_dim`.
  - **Metrics**: Area Under Curve (AUC).
  - **Use Cases**: Detecting fraudulent transactions/account takeovers, ensuring compliance with regional regulations, geolocation-based personalization.
- **K-Means**
  - **Concept**: Groups data into a pre-determined number (`K`) of clusters. It iteratively assigns data points to the nearest cluster centroid (center point) and then recalculates the centroid based on the new members.
  - **Analogy**: At a party, you want to create `K=3` interest groups. You pick 3 random "leaders" (centroids). Guests join the leader they have the most in common with. Then, new leaders are chosen based on the actual average interest of each group, and guests re-evaluate, repeating until the groups are stable.
  - **Problem Type**: Clustering.
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON` for inference.
  - **Compute**: Recommends CPU (GPU supported for single instance only).
  - **Required Hyperparameters**: `feature_dim`, `K`.
  - **Metrics**: Mean Square Distance (msd), Sum of Square Distance (ssd).
  - **Use Cases**: Customer segmentation, market segmentation, recommendation systems.

### 5. Algorithms for Text Data (NLP)
- **Object2Vec**
  - **Concept**: A customizable neural embedding algorithm that creates vector representations (embeddings) of various objects (e.g., sentences, users, products) by learning from their relationships. The goal is to adjust the vectors so that objects with similar relationships are closer together in the embedding space.
  - **Analogy**: A librarian organizes books (objects) by creating a mental map (embedding space) where similar books are placed together. By observing what readers borrow, the librarian refines the map, making it easier to find related books.
  - **Problem Type**: General Purpose Embedding.
  - **Data Formats**: `JSON Lines` (sentence-sentence or label-sentence pairs) for training. `JSON` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `enc0_max_seq_len`, `enc0_vocab_size`.
  - **Metrics**: MSE (Regression); Accuracy, Cross-entropy (Classification).
  - **Use Cases**: User behavior analysis, sentiment analysis, social network analysis.
- **Latent Dirichlet Allocation (LDA)**
  - **Concept**: An unsupervised generative probabilistic model that discovers underlying topics in a collection of documents by analyzing word frequencies.
  - **Analogy**: A librarian organizes a pile of books by guessing genres (topics) and then refining those genres by observing the common words within each book (e.g., "detective," "murder" for the Mystery genre).
  - **Problem Type**: Topic Modeling.
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON` for inference.
  - **Compute**: Supports single-instance CPU only.
  - **Required Hyperparameters**: `num_topics`, `feature_dim`, `mini_batch_size`.
  - **Metrics**: Per-Word-Log-Likelihood (pwll).
  - **Use Cases**: Analyzing customer feedback themes, identifying social media trends, generating content ideas.
- **Neural Topic Model (NTM)**
  - **Concept**: An unsupervised algorithm, similar to LDA, but uses a neural network to model topics. It is more scalable than LDA but can be less interpretable due to the "black box" nature of neural networks.
  - **Problem Type**: Topic Modeling.
  - **Data Formats**: `CSV`, `protobuf` for training. `JSON`, `JSON Lines` for inference.
  - **Compute**: Supports CPU & GPU.
  - **Required Hyperparameters**: `num_topics`, `feature_dim`.
  - **Metrics**: Total loss.
  - **Use Cases**: Uncovering customer pain points, personalized content recommendations, market sentiment analysis.
- **BlazingText**
  - **Concept**: A highly optimized implementation of `Word2Vec` (for creating word embeddings) and `FastText` (for text classification). It is significantly faster than the original implementations, turning days of training into minutes.
  - **Modes**:
    - `word2vec` (unsupervised): cbow, skip-gram, batch_skip-gram.
    - `text classification` (supervised).
  - **Data Formats**: A single preprocessed text file (space-separated tokens). `JSON` for inference.
  - **Compute**: Supports single CPU/GPU; multiple CPUs for batch_skip-gram.
  - **Required Hyperparameters**: `mode`.
  - **Metrics**: `mean_rho` (Word2Vec); Accuracy (Text Classification).
  - **Use Cases**: Sentiment analysis, document classification, recommendation systems.
- **Sequence-to-Sequence (Seq2Seq)**
  - **Concept**: A supervised algorithm that transforms an input sequence to an output sequence using an encoder-decoder neural network architecture. The encoder compresses the input into a feature vector, and the decoder converts that vector into the output sequence.
  - **Problem Type**: Language Processing (Translation, Summarization).
  - **Data Formats**: `protobuf` for training. `JSON`, `protobuf` for inference.
  - **Compute**: Supports single-machine GPU only.
  - **Required Hyperparameters**: None.
  - **Metrics**: Accuracy, BLEU score, Perplexity.
  - **Use Cases**: Machine translation, speech-to-text conversion, code generation.

### 6. Algorithms for Image Data
- **Image Classification**
  - **Concept**: A supervised algorithm that classifies an entire image into one or more categories using a Convolutional Neural Network (CNN).
  - **Modes**:
      - `Full training`: Train from scratch on a large dataset.
      - `Transfer learning`: Fine-tune a pre-trained model on a smaller, specific dataset.
  - **Data Formats**: `recordIO` or image formats (`JPG`, `PNG`), plus a `.lst` file listing images.
  - **Compute**: Recommends GPU for training.
  - **Required Hyperparameters**: `num_classes`, `num_training_samples`.
  - **Metrics**: Accuracy.
  - **Use Cases**: Medical image diagnosis (X-rays), classifying objects for autonomous vehicles, security surveillance.
- **Object Detection**
  - **Concept**: Goes beyond classification to identify and locate *multiple* objects within a single image by drawing bounding boxes around them. It uses the Single Shot MultiBox Detector (SSD) framework.
  - **Data Formats**: `recordIO` or image formats (`JPG`, `PNG`), with a matching `.json` file for each image's annotations.
  - **Compute**: Recommends GPU for training.
  - **Required Hyperparameters**: `num_classes`, `num_training_samples`.
  - **Metrics**: Mean Average Precision (mAP).
  - **Use Cases**: Automated retail checkout systems, manufacturing quality control, scanning information from documents.
- **Semantic Segmentation**
  - **Concept**: The most granular image task, assigning a class label to *every single pixel* in an image to understand object shapes. The output is a segmentation mask (a grayscale image where each shade represents a class).
  - **Data Formats**: Requires separate directories for `train`/`validation` images (`JPG`) and their corresponding `train_annotation`/`validation_annotation` label maps (`PNG`).
  - **Compute**: Recommends GPU for training.
  - **Required Hyperparameters**: `num_classes`, `num_training_samples`.
  - **Metrics**: Mean Intersection-over-Union (mIOU), Pixel Accuracy.
  - **Use Cases**: Analyzing satellite imagery, retail shelf analysis, content moderation in media.

---

## Task Statement 3.3: Model Training & Optimization

### 1. Data Preparation for Training
- **Data Splitting**: The process of splitting data to prevent overfitting and improve performance.
  - **Training Data**: Used for the model to learn hidden patterns.
  - **Validation Data** (Optional): Used during training to measure performance and tune hyperparameters.
  - **Testing Data**: Used after training to see how well the model generalizes to new, unseen data.
- **Cross-Validation**: The process of validating a model against fresh data to get a stable estimate of its performance.
  - **K-Fold Cross-Validation**: The dataset is split into K folds. The model is trained on K-1 folds and tested on the remaining fold. This is repeated K times, and the average error is computed. It's effective but computationally expensive.
  - **Stratified K-Fold**: Similar to K-Fold, but ensures each fold maintains the same class distribution as the entire dataset. Crucial for imbalanced classification problems (e.g., fraud detection).
  - **Leave-One-Out (LOOCV)**: An extreme version where K equals the number of data points. Not often used due to high computational cost.
- **Data Shuffling**: Randomizing the order of data to prevent the model from learning unintended patterns from the data's sequence, which enhances generalization. Many SageMaker built-in algorithms (XGBoost, Linear Learner) do this internally.
- **Bootstrapping**: A statistical technique of creating multiple data samples by resampling from the original dataset *with replacement*. It helps improve model stability and estimate confidence intervals for performance metrics.

### 2. Optimization Techniques
- **Why Optimize?**
  - To minimize the **Loss Function** (the difference between predicted and actual output).
  - To ensure the model **converges** quickly, saving time and resources.
  - To efficiently handle **large datasets** by updating parameters incrementally.
  - To find the right **bias-variance tradeoff**, preventing overfitting and underfitting.
- **Gradient Descent**: A common iterative algorithm used to find the minimum of a function (the loss function).
  - **Learning Rate**: The "step size" taken to reach the minimum error. Too large can overshoot the minimum; too small is computationally expensive.
  - **Convergence**: The stable point reached at the end of the process.
  - **Local vs. Global Minimum**: In non-convex problems, the algorithm can get stuck in a *local minimum*, which isn't the absolute lowest error point.
- **Stochastic Gradient Descent (SGD)**: Updates model parameters using a randomly sampled subset (or single data point) at each iteration. The "noise" introduced helps escape local minima.
- **Batch Gradient Descent**: Updates model parameters only after calculating the gradient from the *entire* dataset. It's more stable but memory-intensive and infeasible for very large datasets.

### 3. Choosing Compute Resources
- **CPUs vs. GPUs**
  - **Model Complexity**: **CPUs** are fine for simple models (Linear/Logistic Regression); **GPUs** are better for complex deep learning models.
  - **Model Size**: **CPUs** for smaller models; **GPUs** for larger models that require more memory.
  - **Inference Needs**: **CPUs** for low-latency needs (e.g., real-time recommendations); **GPUs** for high-throughput batch processing.
  - **Cost**: **CPUs** are cheaper; **GPUs** are more expensive but offer significant performance gains.
- **Recommended EC2 Instances**
  - **General Purpose**: `m5.xlarge`, `m5.4xlarge`
  - **Compute Optimized**: `c5.xlarge`, `c5.2xlarge`
  - **Accelerated Computing (GPU)**: `p3.xlarge`, `p3.8xlarge`, `p4d.24xlarge`
  - **Tool**: Use **Amazon SageMaker Inference Recommender** to find the ideal instance type and count for deployment.
- **Distributed Training**: Training a model across multiple machines to reduce training time for large datasets or complex models.
  - **Data Parallelism**: Used for **large datasets**. The dataset is split across multiple GPUs, each with a full copy of the model. SageMaker provides the **SMDDP** library.
  - **Model Parallelism**: Used for **large models**. The model itself is partitioned across multiple GPUs, and the dataset flows through them. SageMaker provides the **SMP V2** library.
- **Cost Optimization with Spot Training**
  - **Spot Instances**: Unused EC2 capacity offered at up to a 90% discount. AWS can reclaim them with a 2-minute notice.
  - **Checkpoints**: Crucial for spot training. SageMaker saves the model's progress to S3, so if the job is interrupted, it can restart from the last checkpoint.
  - **Configuration**:
    - `use_spot_instances = True`
    - `max_wait`: Max time to wait for a spot instance (must be > `max_run`).
    - `checkpoint_s3_uri`: The S3 location to save checkpoints.

### 4. Debugging & Monitoring
- **Amazon SageMaker Debugger**: A tool to monitor, profile, and debug training jobs in real time.
  - **Features**:
    - **Saves model state** (weights, gradients, biases) at regular intervals.
    - **Detects common issues** like non-converging loss, vanishing gradients, and resource bottlenecks.
    - **Stops training jobs early** if problems are detected, saving time and money.
  - **Built-in Rules**: Use pre-built rules to identify problems automatically.
  - **Alerts**: Integrate with **Amazon CloudWatch** and **SNS** to send notifications when anomalies occur.
  - **Best Practices**: Use the client library for real-time analysis and visualization of system utilization.

### 5. Updating & Retraining Models
- **Importance**: Retraining models with new, fresh data is essential to maintain accuracy and relevance over time.
- **Amazon SageMaker Canvas**: A no-code, drag-and-drop UI for business users to create ML predictions.
  - **Automated ML**: Automatically selects the best algorithm and tunes hyperparameters based on the data.
  - **Retraining**: Models can be retrained easily. Canvas can be configured to **automatically update** a model whenever its associated dataset is refreshed, ensuring predictions are always based on the latest data.

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