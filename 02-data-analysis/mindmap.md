---
mindmap:
  colorFreezeLevel: 1
---

# Certification Exam Refresher

## Task Statement 2.1: Data Handling

### 1. Identify & Handle Missing Data, Corrupt Data, & Stop Words

#### Missing Data
- **Categories**
  - **MCAR (Missing Completely At Random)**
    - Strategies: Mean, Median, KNN Imputation
  - **MAR (Missing At Random)**
    - Strategy: MICE technique
  - **MNAR (Missing Not At Random)**
    - Strategy: Selection models, Shared parameter models
- **Visualization**
  - Python Library: `missingno`
- **Imputation (No one-size-fits-all)**
  - **Categorical Data**: Use mode
  - **Numerical Data**: Use mean, `KNNImputer` library

#### Stop Words
- **Definition**: Most common words with little meaning (articles, prepositions, pronouns)
- **Benefits of Removal**
  - Little negative consequence
  - Reduces dataset size
  - Improves training time & performance
- **Tools**
  - NLTK library
  - spaCy library
- **Customization**: Can append to the default list based on business needs

### 2. Format, Normalize, Augment, & Scale Data

#### Formatting Corrupted Data
- **Causes**: User errors, sync errors, application errors, etc.
- **Tools & Techniques**
  - Python Libraries: NumPy, Pandas
  - Actions: Address spacing, convert case, replace error values
  - Complex Scenarios: Custom Python functions or maps

#### Outliers
- **Definition**: Data point significantly different from others (high or low)
- **Detection**
  - **Z-score**: Formula: $z = \frac{x - \mu}{\sigma}$
  - **Boxplot (IQR)**
    - IQR = Q3 - Q1
    - Min = Q1 - 1.5 * IQR
    - Max = Q3 + 1.5 * IQR
    - Outliers fall outside this range
- **Handling**
  - **Note**: Not always an error; can be valid info
  - **If Invalid**:
    - Replace (consult domain experts)
    - Delete
  - **Minimize Variation**: Use mean, median, or log transformation

### 3. Sufficient Label Data, Mitigation, & Labeling Tools

#### Imbalanced Datasets
- **Definition**: Uneven distribution of a categorical feature
- **Challenge**: Leads to a biased model
- **Mitigation Strategies**
  - Oversampling (has side effects)
  - Undersampling (has side effects)
  - **SMOTE (Synthetic Minority Oversampling Technique)**
    - Industry standard
    - Generates synthetic samples for the minority class

#### Data Labeling
- **Definition**: Assigning meaningful tags/annotations to raw data to provide context

#### Data Labeling Tools (AWS)
- **Amazon SageMaker Ground Truth**
  - Self-serve model
  - Uses human feedback
- **Amazon SageMaker Ground Truth Plus**
  - Managed, turnkey service
  - Leverages AWS expert workforce
  - Delivers high-quality dataset
  - Promoted as 40% cheaper

---

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