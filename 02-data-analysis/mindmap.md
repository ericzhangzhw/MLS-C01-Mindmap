---
title: markmap
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 4
  maxWidth: 300
  spacingHorizontal: 50
---

# Certification Exam Refresher

## The Machine-Learning Lifecycle

### 1. Data Collection
- Raw data is collected but often cannot be used as-is.

### 2. Data Analysis (Descriptive Analysis)
- A subset of statistical analysis.
- Goal: Analyze data, understand patterns & structure, find relationships.
- **Course Focus Area**

### 3. Data Processing (Data Preparation)
- **Goal**: Detect outliers, fill missing values, address inconsistencies to derive meaningful information.
- **Analogy**: Preparing a garden (weeding, adding nutrients) before planting seeds.
- **Course Focus Area**

### 4. Model Building
- Select the right ML algorithm based on data characteristics.
- Train the algorithm with processed data.
- Split data (e.g., 80/20 training/test ratio) to check how well the model scores on new data.
- Iterative process to fine-tune model accuracy.

### 5. Deployment & Monitoring
- Once desired accuracy is achieved, the model is deployed.
- Model makes real-life predictions and is monitored continuously.

---

## Domain 2.1: Data Analysis & Processing Framework

### Data Analysis Phase (Descriptive Statistics)

#### Data Structure
- Gives clues about the data.
- **Too many features**: Can lead to high dimensionality and a complex model.
- **Too few instances**: May lead to inadequate data and a model with low accuracy.

#### Univariate Analysis (Single Feature)
- **Measures of Frequency**:
  - **Frequency**: Number of occurrences of a specific value.
  - **Mode**: The value that occurs most often. A dataset can be multimodal.
- **Measures of Central Tendency**: Finding a central value to summarize the dataset.
  - **Mean**: The average. Can be skewed by outliers.
  - **Median**: The middle value of a sorted dataset.
- **Measures of Variability (Spread/Dispersion)**:
  - **Range**: Difference between max and min value.
  - **Standard Deviation**: Average distance of each value from the mean. Formula: $\sqrt{\frac{\sum(x - \bar{x})^2}{n}}$
  - **Variance**: Average squared deviation from the mean (Standard Deviation squared).
  - **Interquartile Range (IQR)**: Q3 - Q1. Represents the spread of the middle 50%; great for data with outliers.

#### Multivariate Analysis (Relationship between Features)
- Example: Real estate price based on size.
- **Scatterplot**: Excellent tool to visualize the relationship between two attributes.
- **Correlation Matrix**: Shows the relationship between two variables.
  - Values range from -1 (perfect negative) to +1 (perfect positive).
  - **Note**: Not the same as a Confusion Matrix (used for classification model evaluation).

### Data Processing Phase (Data Preparation)

#### Framework
- Address issues from both instance and feature perspectives.
- **Instance (Row/Observation) Techniques**:
  - Removing duplicate instances.
  - Handling outlier data.
  - Imputing missing data.
  - Resampling imbalanced data.
- **Feature (Column) Techniques**:
  - Feature selection.
  - Encoding techniques.
  - Normalizing and scaling.
  - Binning and transforming.

---

## Domain 2.1: In-Depth Topics

### Fixing Data Errors & Formatting
- **Goal**: Address inconsistencies before deriving insights.
- **Common Issues**:
  - **Improper Data Types**: e.g., string/float categorized as 'object'.
  - **Inconsistent Formatting**: Spacing, capitalization, decimal points.
  - **Inconsistent Concepts**: e.g., 'male', 'M', 'm' for the same gender.
- **Implementation (Python/Pandas)**:
  - Convert case: `.str.lower()`
  - Remove spaces: `.str.replace()`, `.str.strip()`
  - Standardize values: Use a map or function with `.apply()`
  - Round numbers: `.round()`

### Handling Missing Data
- **Causes**: User errors, data corruption, import issues, transformation errors.
- **Visualization**: `missingno` is an excellent Python library for this.
- **Categories (Missing Data Mechanism)**:
  - **MCAR (Missing Completely At Random)**: Unrelated to any data.
    - *Strategy*: Mean, Median, KNN Imputation.
  - **MAR (Missing At Random)**: Dependent on other observed data.
    - *Strategy*: MICE (Multivariate Imputation by Chained Equation).
  - **MNAR (Missing Not At Random)**: Related to observed and unobserved data.
    - *Strategy*: Selection Models, Shared Parameter Models.
- **Handling Techniques**:
  - **Drop Rows**: A valid option for large datasets with very few missing values.
  - **Imputation**:
    - **Categorical**: Use the mode (most frequent value).
    - **Numerical**: Use the mean (if no outliers) or median.
    - **KNNImputer (from scikit-learn)**: Uses values of nearest neighbors to impute.
  - **MICE**: Runs multiple imputations to reduce bias but is computationally intensive.

### Handling Outliers
- **Definition**: An observation that differs significantly in value from other data points.
- **Impact**: Skews metrics like the mean; leads to biased models and wrong predictions.
- **Detection Methods**:
  - **Z-score**: Indicates distance from the mean. Formula: $z = \frac{x - \mu}{\sigma}$. Outliers are often considered where $|z| > 3$.
  - **Boxplot (IQR Method)**: Uses Interquartile Range (IQR = Q3 - Q1). Outliers are data points outside the range of $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.
- **Handling Methods**:
  - **Deletion**: Easiest option, but **Warning**: always consult domain experts as some outliers are valid data points.
  - **Transformation**: Apply log transformations to compress the range.
  - **Imputation**: Replace with median or a predetermined threshold value.

### Processing Text Data (Stop Words)
- **Field**: Natural Language Processing (NLP).
- **Challenges**: Context-specific words, sarcasm, slang, misspellings.
- **Stop Words**: Common words with little meaning (e.g., 'a', 'the', 'in', 'is').
  - **Benefits of Removal**: Reduces dataset size, improves training time and performance.
  - **Warning**: Can change meaning if not done carefully (e.g., removing 'not' from 'not happy').
- **Libraries & Process**:
  - **NLTK / SpaCy**: Common NLP libraries.
  - **Process**: Tokenize text (break into words), then filter out stop words.
  - **Customization**: The default stop word list can be customized for specific business needs.

### Resampling Imbalanced Datasets
- **Definition**: A dataset with an uneven distribution of classes (majority vs. minority).
- **Challenges**: Leads to biased models, misleading accuracy metrics.
- **Techniques**:
  - **Weighting**: Increase the weight of the minority class (hyperparameter tuning).
  - **Undersampling**: Remove instances from the majority class (can cause data loss).
  - **Oversampling**: Duplicate instances from the minority class (can cause overfitting).
  - **SMOTE (Synthetic Minority Oversampling Technique)**:
    - Industry standard via `imblearn` library.
    - Generates *synthetic* samples for the minority class by creating new data points along the lines between an instance and its nearest neighbors.
    - Helps avoid overfitting.

### Data Labeling
- **Purpose**: Assigns meaningful tags to raw data, creating a "ground truth" for high-quality ML models.
- **Business Cases**: Image classification, spam detection, medical diagnosis, sentiment analysis.
- **AWS Services**:
  - **Amazon SageMaker Ground Truth**:
    - Self-serve labeling service using human feedback.
    - **Workforce Options**:
      - **Amazon Mechanical Turk**: Public, vendor-managed workforce.
      - **Private Workforce**: Your own team for confidential data.
      - **AI Applications**: Rekognition, Textract, etc.
  - **Amazon SageMaker Ground Truth Plus**:
    - Managed, turnkey service.
    - Leverages an expert workforce managed by AWS.
    - Promoted as a simpler, cost-effective solution for high-quality training datasets.

## Task Statement 2.2: Feature Engineering

### 1. Identify & Extract Features
- **Feature Engineering**: Selection, Extraction, & Transformation
- **Feature Types**
  - **Qualitative**: Nominal, Ordinal, Boolean
  - **Quantitative**: Discrete, Continuous
- **Text Feature Extraction**
  - **Bag of Words**
  - **N-gram**
  - **TF-IDF**: Measures word occurrence & frequency
    - TF: $\frac{\text{Term occurrences in doc}}{\text{Total terms in doc}}$
    - IDF: $\log\left(\frac{\text{Total # of docs}}{\text{# of docs with term} + 1}\right)$
    - Higher value means more significant term
  - **Word Embedding**: Captures context, semantic & syntactic similarity
    - CBoW (Continuous Bag of Words)
    - Skip-gram
    - Stemming
- **Image Feature Extraction**
  - **Traditional CV**
    - Grayscale pixel values (0-255)
    - Mean Pixel Value of Channels (R, G, B)
    - Prewitt Kernels (edge detection)
  - **Deep Learning Techniques**

### 2. Analyze & Evaluate Feature Engineering Concepts
- **Feature Scaling**: Gives same weightage to all features
  - **StandardScaler (Z-score)**: Assumes normal distribution
  - **MinMaxScaler**: Scales data (e.g., between 0 and 1). Formula: $\frac{X - X_{min}}{X_{max} - X_{min}}$
- **Feature Transformation**: Effective against skewed data
  - Strategies: Box-Cox, Polynomial, Exponential
- **Data Binning**: Converts continuous variables to discrete
  - **Equal-width**: $\frac{\text{range}}{\text{# of bins}}$
  - **Equal-frequency**: $\frac{\text{# of elements}}{\text{total # of bins}}$
  - **Other**: K-means binning, Decision tree binning
- **Encoding**: Converts categorical data to numerical
  - **Ordinal Encoding**: For ordinal data
  - **One-Hot Encoding**: For nominal data
  - **Binary Encoding**: For features with many categories
- **Dimensionality Reduction**: Reduces feature count with minimal info loss
  - **Feature Selection**: Selects a subset of features
    - **Filter method**: Filters a subset
    - **Wrapper method**: Uses ML model performance
    - **Embedded method**: Integrated into model training
  - **Feature Extraction**: Transforms features to a lower-dimensional space
    - **Linear**: e.g., PCA (Principal Component Analysis)
    - **Non-linear**
- **Amazon SageMaker Preprocessing**
  - **SKLearnProcessor**: Run scikit-learn scripts as jobs
  - **PySparkProcessor**: Run PySpark scripts as jobs

---

## Task Statement 2.3: Visualization & Analysis

### 1. Create Graphs (Visualization Techniques)
- **Goal**: Understand when to use each technique to tell a story
- **Visualizing Relationships**
  - Uncovers patterns and trends
  - **Scatterplots & Bubble Charts**: Plot features on X/Y axes (Bubble chart for 3+ features)
- **Visualizing Data Distribution**
  - Shows mean, median, skewness
  - **Histogram, Boxplot, Heatmap**
  - Boxplots are effective for identifying outliers
- **Visualizing Comparisons**
  - Static snapshot of how variables compare & change over time
  - **Bar Charts & Line Charts**: Many variations available
- **Visualizing Composition**
  - Shows individual elements of the data
  - **Pie Chart**: Effective for part-to-whole relationships
  - **Stacked Bar Chart**

### 2. Descriptive Statistics
- **Definition**: Collecting & interpreting data (vs. inferential statistics for prediction)
- **Visualization-Specific Metrics**
  - Skewness
  - Kurtosis
  - Correlation

### 3. Cluster Analysis
- **Definition**: Statistical technique to group similar data points
- **Use Case**: When a target variable is not identified
- **Types**
  - **K-Means Clustering (Partition)**
    - Must specify the number of clusters (k)
    - **Elbow Method**: Helps choose an optimal 'k'
  - **Hierarchical Clustering**
    - No need to specify cluster count
    - Forms a **dendrogram** (tree-like structure)
    - **Approaches**
      - **Agglomerative (Bottom-up)**: Start with single points, merge based on similarity.
        - **Linkage Methods**: Calculate distance (e.g., single linkage)
      - **Divisive (Top-down)**: Start with one cluster, split based on dissimilarity.
    - **Note**: Computationally intensive, high memory needs for large datasets.
  - **Density-based Clustering**

### 4. Amazon QuickSight
- **Definition**: AWS BI tool to visualize and analyze data
- **Features**
  - Integrates with multiple data sources
  - Supports many data formats
- **ML Insights**
  - A powerful feature for anomaly detection and forecasting using ML