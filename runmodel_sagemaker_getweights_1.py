# -*- coding: utf-8 -*-
"""RunModel_SageMaker_GetWeights.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rb-6vw-j-3rGNzQwqOM2aTFTlXymKSBv
"""

!pip install sagemaker

"""# **New Code - File :model_revised_claims.py**"""

import sagemaker
from sagemaker.tensorflow import TensorFlow
import boto3
import os


region = 'us-east-2'

# Create a boto3 session with the specified AWS keys and region
boto3_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region
)

# Define the S3 paths
s3_bucket = 'claimsnlp'
s3_folder = 'model_files'

# Local script path after downloading from S3
local_script_path = '/content/model_revised_claims_np.py'
requirements_path = '/content/requirements.txt'

# Define S3 bucket and script path
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region)
s3.download_file(s3_bucket, f'{s3_folder}/model_revised_claims_np.py', local_script_path)
s3.download_file(s3_bucket, f'{s3_folder}/requirements.txt', requirements_path)

# Role with necessary permissions
role = 'arn:aws:iam::471112779710:role/service-role/AmazonSageMaker-ExecutionRole-20240630T134624'  # Replace with your actual role ARN

# Create a SageMaker session
sagemaker_session = sagemaker.Session(boto_session=boto3_session)

# Create an estimator
estimator = sagemaker.tensorflow.TensorFlow(
    entry_point=local_script_path,
    role=role,
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    framework_version='2.3.0',
    py_version='py37',
    script_mode=True,
    hyperparameters={
        'epochs': 200,
        'batch-size': 16
    },
    output_path=f's3://{s3_bucket}/output_files',
    sagemaker_session=sagemaker_session,
    dependencies=[requirements_path]  # Include the requirements.txt file
)

# Define the inputs
inputs = {
    'training': f's3://{s3_bucket}/{s3_folder}/Claims_Raw_Data.csv'
}

# Start the training job
estimator.fit(inputs)
