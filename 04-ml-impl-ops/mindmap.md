---
title: Enhanced AWS ML Module Review
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 300
---

# Enhanced AWS ML Module Review

- ## 1. ML Implementation & Operations
  - ### Model Deployment Strategies
    - #### **SageMaker Hosted Endpoints** (Real-time Inference)
      - The easiest way to deploy a model.
      - Creates a persistent HTTPS endpoint for on-demand predictions.
      - **How it works**: Deploys a model to a container on **ECS**, running on **EC2**. Fully managed by SageMaker.
      - **Deployment**: Use the `deploy()` method in the Python SDK.
    - #### **SageMaker Serverless Inference**
      - **Use Case**: For workloads with intermittent or unpredictable traffic and idle periods.
      - **Benefit**: Pay-per-use, automatically scales compute resources. Can tolerate cold starts.
    - #### **SageMaker Batch Transform**
      - **Use Case**: Run inference on an entire dataset at once (offline).
      - **Benefit**: No persistent endpoint needed. Manages provisioning and de-provisioning of resources.
      - Can also be used to pre-process datasets or associate input records with inference results.
    - #### **Manual Deployment Options**
      - **EC2 Instance**: Use **Deep Learning AMIs** (pre-configured with frameworks like TensorFlow, PyTorch) and select a powerful instance type (e.g., P5 with GPUs).
      - **Custom Container on ECS**: Build your own Docker container and deploy it on Elastic Container Service for full control.
  - ### Scalability & Resiliency
    - #### **High Availability & Fault Tolerance**
      - **Goal**: Avoid single points of failure. Design for failure ("Everything fails all the time").
      - **Method**: Deploy resources to multiple locations.
        - Multiple **Availability Zones (AZs)** to protect against instance/AZ failure.
        - Multiple **Regions** to protect against regional failure.
      - **Best Practice**: Use a greater number of smaller instance types instead of fewer larger ones to minimize the impact of a single instance failure.
      - **Reference**: AWS Well-Architected Framework.
    - #### **Auto Scaling Model Endpoints**
      - **How it works**: Automatically adjusts the number of instances in response to traffic.
      - **Configuration**: Define a scaling policy based on a metric, such as `SageMakerVariantInvocationsPerInstance`.
      - SageMaker automatically load balances requests across all instances.
    - #### **Multi-Model Endpoints**
      - **Use Case**: Host many models that use the same ML framework on one shared container.
      - **Benefits**:
        - **Cost-Effective**: Improves container utilization, avoiding a new endpoint for each model.
        - **Reduced Overhead**: Manage one endpoint instead of many.
  - ### Model Testing in Production
    - #### **A/B Testing (Production Variants)**
      - **Method**: Distribute traffic to multiple model versions on the same endpoint.
      - **Configuration**: Assign a **weight** to each **production variant** (e.g., 20% to model A, 80% to model B).
      - **Updating**: Weights can be updated without any application code changes.
      - **Direct Invocation**: Call a specific model version using the `TargetVariant` parameter.

- ## 2. Security & Identity
  - ### Identity & Access Management (IAM)
    - **Principle of Least Privilege**: Grant only the permissions an entity (user, role, service) needs to perform its function.
    - **SageMaker Execution Roles**: An IAM role that grants SageMaker permission to perform operations on your behalf (e.g., reading data from S3, writing model artifacts).
      - The default `AmazonSagemakerFullAccess` policy grants broad permissions; it's best practice to create more restrictive, custom policies.
  - ### Data & Network Protection
    - #### **Amazon S3 Security**
      - **Default**: All S3 buckets are **private**.
      - **Bucket Policies**: Define fine-grained access permissions for users and roles.
      - **Conditions**: Can be used in policies to enforce rules, like requiring HTTPS.
    - #### **Encryption with AWS KMS**
      - **How it works**: KMS uses **Envelope Encryption** (a Customer Master Key encrypts a Data Key, which encrypts your data).
      - **Key Policies**: Control which users and roles can use KMS keys.
      - **Critical for SageMaker**: If you use a customer-managed KMS key for S3 data, the SageMaker Execution Role **must** have `encrypt` and `decrypt` permissions in the key's policy.
    - #### **VPC & Private Communication**
      - **Virtual Private Cloud (VPC)**: Your logical, isolated data center in AWS.
      - **VPC Endpoints (AWS PrivateLink)**
        - Enables **private and secure communication** between your VPC and AWS services (like SageMaker and S3).
        - Traffic **never leaves the Amazon network**; no internet gateway, public IPs, or public subnets are needed.
  - ### Other Security Concepts
    - **Data Anonymization**: Techniques like hashing values with Amazon Athena to protect sensitive data that cannot be removed.
    - **Security Groups**: A stateful virtual firewall for EC2 instances that controls inbound and outbound traffic.

- ## 3. MLOps, Monitoring & Governance
  - ### Monitoring vs. Auditing
    - #### **Amazon CloudWatch (Monitoring)**
      - **Purpose**: Monitors the **performance and health** of systems and applications.
      - **Metrics**: Collects near-real-time utilization metrics (CPU, Memory, GPU) for SageMaker instances.
      - **CloudWatch Logs**: Centralizes, monitors, and stores log files.
      - **CloudWatch Events (EventBridge)**: Responds to system events and state changes (e.g., a training job completing).
      - **CloudWatch Alarms**: Sends notifications (via SNS/SES) when a metric breaches a defined threshold.
    - #### **AWS CloudTrail (Auditing)**
      - **Purpose**: Provides a **governance and audit trail** of actions in your AWS account.
      - **What it logs**: API calls made by users, roles, or services (e.g., `CreateTrainingJob`, `CreateModel`, `CreateEndpoint`).
      - **Long-term storage**: Create a **trail** to save logs indefinitely to an S3 bucket.
  - ### Advanced Customization & Automation
    - #### **Custom Docker Containers**
      - **Use Case**: When a pre-built SageMaker container doesn't fit your needs (e.g., custom algorithm, specific library versions).
      - **Process**:
        1.  Write a `Dockerfile`.
        2.  Build the Docker image.
        3.  Push the image to **Amazon Elastic Container Registry (ECR)**.
        4.  Use the ECR image URI in your SageMaker training or deployment job.
    - #### **Golden Images (AMIs)**
      - **What It Is**: A pre-configured Amazon Machine Image from an existing EC2 instance.
      - **Includes**: OS, software, packages, scripts, and configurations.
      - **Benefit**: Ensures consistent configuration, avoids manual setup, and is used with Auto Scaling.
    - #### **Retraining Pipelines with AWS Step Functions**
      - **Purpose**: Orchestrate and automate the entire ML workflow (data pre-processing, training, evaluation, deployment).
      - **Benefits**: Visualizes the pipeline, logs each step's state for debugging, and manages dependencies.
  - ### SageMaker Model Monitor
    - **Purpose**: Continuously monitor a deployed model for drift.
    - **Detects Drift In**:
      - **Data Quality**: Changes in the production data distribution.
      - **Model Quality**: Degradation in performance metrics (e.g., accuracy).
      - **Bias**: Introduction of bias as data changes.
      - **Feature Attribution**: Changes in feature importance.

- ## 4. AWS AI/ML Service Landscape
  - ### The AWS Machine Learning Stack (3 Layers)
    - #### **Bottom Layer: Infrastructure & Frameworks**
      - **For**: ML experts wanting full control.
      - **Services**: EC2 (w/ GPUs), ECS, Deep Learning AMIs, Docker.
    - #### **Middle Layer: Amazon SageMaker**
      - **For**: Developers and data scientists wanting to streamline the ML lifecycle.
    - #### **Top Layer: AI/ML Services**
      - **For**: Customers who want to add intelligence to apps without building/training models.
  - ### High-Level AI Service Details
    - **Textract**: Extracts text/data from documents.
    - **Transcribe**: Speech-to-Text.
    - **Translate**: Language translation.
    - **Comprehend**: NLP for sentiment, key phrases, topics.
    - **Lex**: Builds conversational chatbots.
    - **Polly**: Text-to-Speech.
    - **Rekognition**: Image and video analysis.
    - **Forecast**: Time-series forecasting.
    - **Fraud Detector**: Online fraud detection.