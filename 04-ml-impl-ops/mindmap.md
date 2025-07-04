---
title: markmap
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 4
  maxWidth: 300
---

# Comprehensive AWS ML Module Review

- ## Module Review: Deployment, Services & ML Stack
  - ### Model Deployment Options
    - #### SageMaker Endpoint (Easiest Way)
      - Deploys model to a container on ECS, running on EC2.
      - Fully managed by **SageMaker**.
    - #### EC2 Instance
      - Use **Deep Learning AMIs**.
      - Select a **Deep Learning instance type** (e.g., P5).
    - #### Custom Container on ECS
      - Build and run it yourself in Elastic Container Service.
  - ### Access Control & Security
    - #### Principle of Least Privilege
      - Grant only the permissions needed to perform a role.
      - SageMaker supports **Execution Roles** for this.
    - #### S3 Data Protection
      - Use **encryption**.
      - Use properly configured **bucket policies**.
    - #### Using KMS Encryption
      - If using customer-managed KMS keys on S3:
        - The role accessing the data **must** have `encrypt` and `decrypt` permissions for that specific KMS key.
  - ### High Availability & Fault Tolerance
    - **Goal**: Avoid single points of failure.
    - **Method**: Deploy to multiple locations.
      - Multiple **Availability Zones (AZs)**.
      - Multiple **Regions**.
    - A single component/AZ/region failure should not bring the system down.
  - ### VPC Endpoints (Private Communication)
    - Enable private communication between your VPC and AWS services (SageMaker, S3).
    - No public IPs, public subnets, or internet gateways are needed.
    - Uses **AWS PrivateLink**, so traffic never leaves the Amazon network.
  - ### Multi-Model Endpoints
    - **Use Case**: Host many models using the same ML framework on one shared container.
    - **Benefits**:
      - **Cost-Effective**: Better container utilization, avoids deploying a new endpoint for each model.
      - **Reduced Admin Overhead**: Only one endpoint to manage.
  - ### AMIs (Golden Images)
    - **What It Is**: A preconfigured image from an existing EC2 instance.
    - **Includes**: OS, software, packages, scripts, and configurations.
    - **Benefit**: Ensures consistent configuration and avoids manual setup steps.
  - ### Monitoring: CloudWatch vs. CloudTrail
    - #### Amazon CloudWatch (Monitoring Systems & Apps)
      - Collects **near-real-time utilization metrics** (CPU, Memory, GPU).
      - **CloudWatch Logs**: Collects system and application logs.
      - **CloudWatch Events** (EventBridge): Notifies on status changes.
      - **CloudWatch Alarms**: Alerts and sends notifications (via SNS/SES) when a threshold is breached.
    - #### AWS CloudTrail (Auditing Actions)
      - Provides an **audit trail** of actions in your AWS account.
      - Logs actions/API calls by users, roles, or services.
      - **Examples**: `CreateTrainingJob`, `CreateModel`, `CreateEndpoint`.
      - **To store logs indefinitely**: Create a **trail** to save logs to an S3 bucket.
  - ### The AWS Machine Learning Stack (3 Layers)
    - #### Bottom Layer: Infrastructure & Frameworks
      - **For**: ML experts wanting full control.
      - **Services**: EC2 (w/ GPUs), ECS, Deep Learning AMIs, Docker.
    - #### Middle Layer: Amazon SageMaker
      - **For**: Removing the heavy lifting in the ML lifecycle.
    - #### Top Layer: AI/ML Services
      - **For**: Customers who don't want to build/train models.
      - **Services**: Textract, Transcribe, Translate, Lex, Polly, Rekognition, Comprehend.
  - ### Service Quotas (Limits)
    - Limits exist on services (e.g., number of EC2 instances).
    - Can impact SageMaker jobs.
    - **Management**:
      - Request quota increases in the AWS Console.
      - Check limits with **Trusted Advisor**.
  - ### SageMaker: Built-in vs. Custom Models
    - SageMaker offers many **built-in algorithms**.
    - If a model isn't available, you can build your own **Docker image**.
  - ### Distinguishing AI Services
    - **Textract**: Extracts text/data from documents (invoices, IDs).
    - **Transcribe**: Speech-to-Text.
    - **Translate**: Language translation.
  - ### High-Level AI Service Details
    - **Comprehend**: Understands text data using NLP (sentiment, key phrases, topics).
    - **Lex**: Builds chatbots for conversational experiences.
    - **Polly**: Text-to-Speech, creates natural-sounding audio.
    - **Forecast**: For demand forecasting using time-series data.
    - **Rekognition**: Image and video analysis (objects, text, identity verification).
    - **Fraud Detector**: Low-latency online fraud protection based on your historical data.

- ## Module Review: Security & Identity
  - ### Identity & Access Management (IAM)
    - **Users**: People
    - **Groups**: A collection of users with one set of permissions.
    - **Roles**: Can be assigned to users, applications, or services for access.
    - **Policies**: Documents defining permissions, attached to a user, group, or role.
      - Can include **conditions** to limit access.
  - ### Amazon S3 Security
    - **Default**: All newly created buckets are **private**.
    - **Bucket Policies**: Used to allow and deny access to specific IAM roles/users.
    - **Conditions**: Can be used in policies to enforce rules, like requiring HTTPS.
  - ### Virtual Private Cloud (VPC)
    - **Concept**: A logical, customizable data center in AWS.
    - **Components**: Subnets, Route Tables, NACLs, Security Groups, Internet Gateways.
    - **Subnets**:
      - **Public**: Has a route to an Internet Gateway for public access.
      - **Private**: No Internet Gateway; may have outbound internet via a **NAT Gateway**.
  - ### Security Groups
    - **Concept**: A **stateful** virtual firewall for EC2 instances.
    - **Default State**: Denies all inbound traffic. You create `allow` rules.
    - **Rule Components**: Protocol, Port Range, Source/Destination.
    - **Source/Destination**: Can be an IP, a network range, or another Security Group ID.
  - ### Encryption
    - **Purpose**: Protect data from unauthorized access.
    - **AWS KMS (Key Management Service)**
      - Uses **Envelope Encryption**:
        1. A **Customer Master Key (CMK)** encrypts...
        2. a **Data Key**, which then...
        3. encrypts your actual data.
      - **Key Policies**: Control which users and roles can use the keys to encrypt/decrypt.
  - ### Data Anonymization
    - **Use Case**: For sensitive data when fields cannot be removed.
    - **Example Method**: Use **Amazon Athena** to run a SQL query that creates a hash of the original values, disguising them.

- ## Module Review: Model Testing & Monitoring
  - ### A/B Testing in Production
    - **Method**: Distribute traffic to multiple **production variants** on the same endpoint.
    - **Configuration**:
      - Specify a **weight** for each variant in the endpoint configuration (e.g., `0.2` for 20%, `0.8` for 80%).
      - SageMaker automatically routes requests based on the weights.
    - **Updating**:
      - Update weights in the endpoint configuration.
      - **No application code change is needed.**
    - **Direct Invocation**:
      - Invoke a specific model version by specifying the `TargetVariant` in the `invoke_endpoint` API call.
  - ### Retraining Pipelines
    - **Scheduling**: Can be scheduled using **AWS Step Functions**.
    - **Benefits of Step Functions**:
      - Visualize the ML pipeline.
      - Automatically trigger and track each step.
      - Output of one step becomes the input to the next.
      - Logs the state of each step for easier debugging.
  - ### SageMaker Model Monitor
    - **Purpose**: Monitor the performance and effectiveness of a deployed model.
    - **Detects Drift In**:
      - **Data Quality**: Production data distribution changes from training data.
      - **Model Quality**: Metrics like accuracy or precision degrade.
      - **Bias**: Bias is introduced as production data changes.
      - **Feature Attribution**: The relative importance of features changes.