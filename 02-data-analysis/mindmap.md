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

## Task Statement 2.2: Perform Feature Engineering

### 1. What is Feature Engineering?
- **Definition**: Selecting, extracting, & transforming variables from raw data to effectively train a model.
- **Core Concepts**
  - **Feature Selection**
    - **Definition**: Choosing a subset of relevant features.
    - **Goals**: Improve performance, reduce computational cost, enhance interpretability.
    - **Example**: Removing 'Owner Name' for house price prediction.
  - **Feature Extraction**
    - **Definition**: Creating new features from existing ones.
    - **Goals**: Provide more relevant info, reduce dimensionality.
    - **Example**: Creating 'cost per sq ft' from 'price' & 'total sq ft'.
  - **Feature Transformation**
    - **Definition**: Converting features to a more suitable representation for models.
    - **Goals**: Mitigate data skewness, handle differing scales.
    - **Example**: Log transform on house prices.
- **Challenges**
  - Requires deep **domain understanding**.
  - Time-consuming and resource-intensive process.
  - No one-size-fits-all approach; requires mastery of various techniques.
- **Value Proposition**
  - Mitigates risk of overfitting and improves generalization.
  - Captures underlying data patterns for better predictions.
  - Helps business stakeholders make better decisions.

### 2. Identifying Feature Types
- **Qualitative (Categorical)**
  - **Nominal**: No inherent order (e.g., heating type: gas, electric).
  - **Ordinal**: Clear order or ranking (e.g., house quality: poor, fair, good).
  - **Boolean**: Binary value (e.g., on sale: yes/no).
- **Quantitative (Numerical)**
  - **Discrete**: Countable items (e.g., # of bedrooms).
  - **Continuous**: Infinite values within a range (e.g., price, sq ft).

### 3. Feature Engineering Techniques
- #### In-Depth: Text Feature Extraction
  - **Goal**: Convert raw text into a structured, numerical format.
  - **Preprocessing is crucial**: Lowercase, remove punctuation & stop words.
  - **Techniques**
    - **Bag of Words (BoW)**
      - **Process**: Tokenizes text and measures frequency of words in a document.
      - **Key Trait**: Ignores word order and sequence.
      - **Scoring**: Binary (presence), Count, or Frequency.
      - **Limitation**: High dimensionality for large vocabularies.
    - **N-gram**
      - **Process**: Builds on BoW to capture groups of 'N' consecutive words.
      - **Unigram (N=1)** is the same as BoW.
      - **Bigram (N=2)** considers pairs of words.
    - **Limitations of BoW & N-gram**
      - Cannot capture context or semantic meaning.
      - Suffer from out-of-vocabulary words.
      - Can be computationally expensive.
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**
      - **What is it?**: Statistical measure of a word's importance to a document in a corpus.
      - **Formula**: TF-IDF Score = TF * IDF
      - **TF**: $\frac{\text{Term occurrences in doc}}{\text{Total terms in doc}}$
      - **IDF**: $\log\left(\frac{\text{Total \# of docs}}{\text{\# of docs with term} + 1}\right)$
      - **High Score**: Word is important to that specific document.
      - **Low Score**: Word is common across many documents (less important).
    - **Word Embedding**
      - **What is it?**: Represents words as dense numerical vectors, capturing context, semantic & syntactic similarity.
      - **Techniques**
        - **CBoW (Continuous Bag of Words)**: Predicts a target word from its context.
        - **Skip-gram**: Predicts context words from a target word. Effective for word similarity tasks.
    - **Stemming**
      - **What is it?**: Reduces words to their root form (e.g., "learning" -> "learn").
      - **Purpose**: Normalizes words to reduce the number of unique words.
- **Image Feature Extraction**: Traditional CV, Deep Learning.
- #### In-Depth: Feature Scaling
  - **Why?** To prevent models from being influenced by features with larger scales (e.g., salary vs. age).
  - **Techniques**
    - **StandardScaler (Z-score)**
      - Scales data to a mean of 0 and std dev of 1.
      - Formula: $z = \frac{x - \mu}{\sigma}$
      - **Assumption**: Data is normally distributed.
    - **MinMaxScaler**
      - Rescales data to a specific range (default is 0 to 1).
      - Formula: $\frac{X - X_{min}}{X_{max} - X_{min}}$
    - **MaxAbsScaler**
      - Scales data to a range of -1 to 1 by dividing by the max absolute value.
    - **RobustScaler**
      - Removes median and scales using the Interquartile Range (IQR).
      - **Recommended for data with outliers**.
- #### In-Depth: Feature Transformation
  - **Why?** To interpret non-linear relationships, reduce skewness, and uncover hidden patterns.
  - **Techniques**
    - **Logarithmic Transformation**
      - Reduces the impact of very high values and makes data less skewed.
    - **Box-Cox Transformation**
      - Used when the target variable is skewed to convert it to a normal distribution.
    - **Polynomial Transformation**
      - Creates new features by raising original features to a power to capture non-linear relationships.
    - **Exponential Transformation**
      - Used for features exhibiting exponential growth or decay (e.g., stock prices).
- #### In-Depth: Data Binning (Discretization)
  - **What is it?** Transforming a continuous variable into discrete bins/intervals.
  - **Benefits**
    - Can improve performance for some ML algorithms (faster training).
    - Reduces noise by grouping similar values.
    - Helps address data skewness.
  - **Drawbacks**
    - **Information Loss**: Fine-grained details can be lost.
    - **Bias**: Inappropriate bin sizes can introduce bias.
  - **Binning Strategies**
    - **Equal-Width (Uniform)**
      - Bins have the same width. Formula: $\frac{\text{range}}{\text{\# of bins}}$
      - Best for symmetric, evenly distributed data.
    - **Equal-Frequency (Quantile)**
      - Bins have the same number of observations.
      - Best for skewed datasets or those with outliers.
    - **K-Means Binning**
      - Uses K-Means clustering to partition data.
      - Good for non-uniform data distributions.
    - **Decision-Tree Binning**
      - Uses a decision tree to find optimal split points.
      - Effective for non-linear relationships.
  - **Implementation**: `scikit-learn`'s **KBinDiscretizer**.
- #### In-Depth: Encoding Categorical Data
  - **What is it?** Transforming categorical data into numerical data.
  - **Importance**
    - **Algorithm Compatibility**: Most ML models require numerical input.
    - **Improve Model Quality**: Helps algorithms learn patterns.
    - **Prevent Bias**: Ensures equal feature weightage.
  - **Encoding Ordinal Features** (Data with clear order)
    - **OrdinalEncoder**: Assigns integers respecting the order (e.g., `Developer:1`, `Manager:3`).
    - **LabelEncoder**: Also assigns integers, but does not guarantee order (encodes alphabetically). Order can be manually set.
  - **Encoding Nominal Features** (Data with no order)
    - **One-Hot Encoding**
      - Creates a new binary (0/1) column for each category.
      - **Best for**: Features with a low number of unique categories.
      - **Drawbacks**: Can lead to high dimensionality and sparse data.
    - **Binary Encoding**
      - Converts categories to integers, then to binary digits, then splits binary digits into columns.
      - **Best for**: Features with a high number of unique categories (high cardinality).
- **Dimensionality Reduction**: Feature Selection (Filter, Wrapper, Embedded), Feature Extraction (PCA).

### 4. Amazon SageMaker Preprocessing
- **SKLearnProcessor**: Run scikit-learn scripts as processing jobs.
- **PySparkProcessor**: Run PySpark scripts as processing jobs.

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