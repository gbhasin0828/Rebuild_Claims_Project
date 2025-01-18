# -*- coding: utf-8 -*-
"""model_revised_claims_np.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10Lnkc-DwwK_OYhJ6RnsrGCIJJ5Mc9KaP
"""

# -*- coding: utf-8 -*-
"""Model_Revised_Claims.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xp16dvPf-GJMREYfUO8zc4Lid0_txETm
"""

#!pip install boto3  # For Amazon S3 access
#!pip install psycopg2-binary  # For PostgreSQL access

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertModel

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

import os


import boto3
import pandas as pd
from io import StringIO

# S3 bucket and file keys
bucket_name = 'claimsnlp'

# Local file paths
local_input_file_path = '/tmp/Claims_Raw_Data_predict_input.csv'
local_output_file_path = '/tmp/Claims_Raw_Data_predict_output.csv'
local_model_file_path = '/tmp/saved_model.h5'


# Function to read CSV from S3
def read_csv_from_s3(bucket, file_key, dtype_dict):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(csv_content), dtype=dtype_dict)


file_key = 'model_files/Claims_Raw_Data.csv'

# Data types dictionary
dtype_dict = {
    'product': 'str',
    'category': 'str',
    'tdp' : 'float',
    'subcategory': 'str',
    'supercategory': 'str',
    'segment' : 'str',
    'manufacturer': 'str',
    'sales': 'float',
    'time': 'str',
    'upc' : 'int'
}

# Read CSV from S3
data = read_csv_from_s3(bucket_name, file_key, dtype_dict)

scaler = MinMaxScaler()
data[['tdp', 'sales']] = scaler.fit_transform(data[['tdp', 'sales']])

# Split the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mark train and test rows
data['train_test_split'] = 'train'
data.loc[X_test.index, 'train_test_split'] = 'test'

X = data.drop(['sales', 'upc', 'supercategory'],axis=1)
y = data['sales']

# Transform DatFrame to EagerTensors
X_train_tdp_input = tf.constant(X_train['tdp'])
X_test_tdp_input = tf.constant(X_test['tdp'])
y_train_input = tf.constant(y_train)
y_test_input = tf.constant(y_test)

# Change the shape of the EagerTensors
X_train_tdp_input1 = X_train_tdp_input[:, tf.newaxis]
X_test_tdp_input1 = X_test_tdp_input[:, tf.newaxis]
#Same Thing 
y_train_input1 = y_train_input[:, tf.newaxis]
y_train_input1 = tf.convert_to_tensor(y_train.values, dtype=tf.float32)[:, tf.newaxis]
y_test_input1 = tf.convert_to_tensor(y_test.values, dtype=tf.float32)[:, tf.newaxis]

#Shange the tensor shape (No Need)
y_test_input = tf.reshape(y_train_input, (-1, 1))
y_train_input = tf.reshape(y_train_input, (-1, 1))


"""# **Handle text variable "product"**"""

# BERT tokenization and encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def encode_product_descriptions(descriptions):
    inputs = tokenizer(descriptions.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

# Convert Product variable to BERT Embeddings
product_embeddings_train = encode_product_descriptions(X_train['product'])
product_embeddings_test = encode_product_descriptions(X_test['product'])

# Combine Embeddings & Attention Masks (No Need)
product_input_train = Input(shape=(product_embeddings_train.shape[1],), name='product')
product_input_test = Input(shape=(product_embeddings_test.shape[1],), name='product')

# Method 2: Using tf.newaxis
product_embeddings_train1= product_embeddings_train[:, tf.newaxis] # Should print (199, 1)
product_embeddings_test1= product_embeddings_test[:, tf.newaxis]


# Final Transormation (Convert to float32)
product_embeddings_train = tf.cast(product_embeddings_train, tf.float32)
product_embeddings_test = tf.cast(product_embeddings_test, tf.float32)


"""# **Handle other categorical variables**"""

# One-hot encode categorical variables
categorical_features = ['time', 'category', 'subcategory', 'segment', 'manufacturer']
onehot_encoder = OneHotEncoder(sparse=False)

categorical_encoded_train = onehot_encoder.fit_transform(X_train[categorical_features])
categorical_encoded_test = onehot_encoder.transform(X_test[categorical_features])

# No need for this
categorical_encoded_df_train = pd.DataFrame(categorical_encoded_train, columns=onehot_encoder.get_feature_names_out(categorical_features))
categorical_encoded_df_test = pd.DataFrame(categorical_encoded_test, columns=onehot_encoder.get_feature_names_out(categorical_features))


# Convert to Eager Tensor
categorical_input_train = tf.constant(categorical_encoded_train)
categorical_input_test = tf.constant(categorical_encoded_test)


# Convert to *** Tensor (No Need)
categorical_input_train1= categorical_input_train[:, tf.newaxis]
categorical_input_test1= categorical_input_test[:, tf.newaxis]


# Final Transormation (Convert to float32)
categorical_input_train = tf.cast(categorical_input_train, tf.float32)
categorical_input_test = tf.cast(categorical_input_test, tf.float32)

X_train_tdp_input1 = tf.cast(X_train_tdp_input1, tf.float32)
X_test_tdp_input1 = tf.cast(X_test_tdp_input1, tf.float32)

y_train_input = tf.cast(y_train_input, tf.float32)
y_test_input = tf.cast(y_test_input, tf.float32)


"""# **Model Building "Putting it all together"**"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

class SequentialModel(tf.keras.Model):
    def __init__(self, output_dim):
        super(SequentialModel, self).__init__()
        self.product_dense1 = Dense(128, activation='relu')
        self.product_dense2 = Dense(64, activation='relu')
        self.product_dense3 = Dense(32, activation='relu')
        self.product_dense4 = Dense(16, activation='relu')

        self.categorical_dense1 = Dense(128, activation='relu')
        self.categorical_dense2 = Dense(64, activation='relu')
        self.categorical_dense3 = Dense(32, activation='relu')
        self.categorical_dense4 = Dense(16, activation='relu')

        self.tdp_dense1 = Dense(128, activation='relu')
        self.tdp_dense2 = Dense(64, activation='relu')
        self.tdp_dense3 = Dense(32, activation='relu')
        self.tdp_dense4 = Dense(16, activation='relu')

        self.dropout = Dropout(0.5)
        self.output_layer = Dense(output_dim)

    def call(self, inputs):
        product_inputs, categorical_inputs, tdp_inputs = inputs

        # Process product embeddings
        product_x = self.product_dense1(product_inputs)
        product_x = self.product_dense2(product_x)
        product_x = self.product_dense3(product_x)
        product_x = self.product_dense4(product_x)
        product_x = self.dropout(product_x)

        # Process categorical inputs
        categorical_x = self.categorical_dense1(categorical_inputs)
        categorical_x = self.categorical_dense2(categorical_x)
        categorical_x = self.categorical_dense3(categorical_x)
        categorical_x = self.categorical_dense4(categorical_x)
        categorical_x = self.dropout(categorical_x)

        # Process tdp input
        tdp_x = self.tdp_dense1(tdp_inputs)
        tdp_x = self.tdp_dense2(tdp_x)
        tdp_x = self.tdp_dense3(tdp_x)
        tdp_x = self.tdp_dense4(tdp_x)
        tdp_x = self.dropout(tdp_x)

        # Combine their representations at the final layer
        combined_x = product_x + categorical_x + tdp_x
        output = self.output_layer(combined_x)
        return output

# Instantiate the model
output_dim = 1  # Change this as needed
model = SequentialModel(output_dim)

output_dim = 1
model = SequentialModel(output_dim)
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

"""# **Inputs**"""


# Prepare input data
input_data_train = [product_embeddings_train, categorical_input_train, X_train_tdp_input1]


# Train the model
model.fit(input_data_train, y_train, epochs=200, batch_size=16)

#combined_x = tf.concat([product_embeddings_train, categorical_input_train, X_train_tdp_input1], axis=1)  (Not using as this is an alternate method to input data)

# Make predictions
y_pred = model.predict([product_embeddings_train, categorical_input_train, X_train_tdp_input1])

# Convert tensors to numpy arrays for scikit-learn

y_train_np = y_train.values

# Compute metrics
mse = mean_squared_error(y_train_np, y_pred)
mae = mean_absolute_error(y_train_np, y_pred)
r2 = r2_score(y_train_np, y_pred)

print(f"MSE train: {mse}")
print(f"MAE train: {mae}")
print(f"R² score train: {r2}")


# Make predictions
y_pred_test = model.predict([product_embeddings_test, categorical_input_test, X_test_tdp_input1])
#y_pred_test = model.predict([product_embeddings_test, categorical_input_test, X_test_tdp_input])


y_test_np = y_test.values

# Compute metrics
mse = mean_squared_error(y_test_np, y_pred_test)
mae = mean_absolute_error(y_test_np, y_pred_test)
r2 = r2_score(y_test_np, y_pred_test)

print(f"MSE test: {mse}")
print(f"MAE test: {mae}")
print(f"R² score test: {r2}")


#####################################################################

import tensorflow as tf
import boto3
import tempfile
import os


# Replace with your AWS credentials
aws_access_key_id = 
aws_secret_access_key = 

# Configure boto3 client with your credentials and region
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name='us-east-2')  # Update with your S3 bucket's region

# Function to upload file to S3
def upload_to_s3(local_file, bucket, s3_file):
    try:
        s3_client.upload_file(local_file, bucket, s3_file)
        print(f"File '{s3_file}' uploaded successfully to '{bucket}'")
    except Exception as e:
        print(f"Failed to upload {local_file} to {bucket}. Error: {e}")

# Save model weights to a temporary file
with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
    temp_file_path = temp_file.name
    model.save_weights(temp_file_path)
    print(f"Model weights saved locally at {temp_file_path}")


# Set your AWS S3 bucket and weights file name
bucket_name = 'claimsnlp'
s3_weights_key = 'model_output/model_weights_np.h5'  # S3 object key (path + filename)

# Upload weights file to S3
upload_to_s3(temp_file_path, bucket_name, s3_weights_key)

# Save the entire model to the TensorFlow SavedModel format locally
local_model_dir = '/tmp/saved_model_np'
model.save(local_model_dir, save_format='tf')
print(f'Model saved locally at {local_model_dir}')


# Upload the model to S3
s3_model_path = 'model_output/saved_model'
for root, dirs, files in os.walk(local_model_dir):
    for file in files:
        local_file_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_file_path, local_model_dir)
        s3_file_path = os.path.join(s3_model_path, relative_path)
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_file_path}")
        upload_to_s3(local_file_path, bucket_name, s3_file_path)



import tensorflow as tf
import boto3
import tempfile
import os
import pandas as pd





# Configure boto3 client with your credentials and region
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name='us-east-2')  # Update with your S3 bucket's region

# Function to upload file to S3
def upload_to_s3(local_file, bucket, s3_file):
    try:
        s3_client.upload_file(local_file, bucket, s3_file)
        print(f"File '{s3_file}' uploaded successfully to '{bucket}'")
    except Exception as e:
        print(f"Failed to upload {local_file} to {bucket}. Error: {e}")

# Convert tdp inputs to 2D tensors
X_train_tdp_input = tf.constant(X_train['tdp'], dtype=tf.float32)[:, tf.newaxis]
X_test_tdp_input = tf.constant(X_test['tdp'], dtype=tf.float32)[:, tf.newaxis]


# Make predictions
y_pred_train = model.predict([product_embeddings_train, categorical_input_train, X_train_tdp_input])
y_pred_test = model.predict([product_embeddings_test, categorical_input_test, X_test_tdp_input])

# Convert predictions to numpy arrays
y_pred_train = y_pred_train.flatten()
y_pred_test = y_pred_test.flatten()

# Create a DataFrame with the predictions
train_results = pd.DataFrame({
    'input_index': X_train.index,
    'true_values': y_train,
    'predicted_values': y_pred_train,
    'train_test_split': 'train'
})

test_results = pd.DataFrame({
    'input_index': X_test.index,
    'true_values': y_test,
    'predicted_values': y_pred_test,
    'train_test_split': 'test'
})

# Concatenate train and test results
results = pd.concat([train_results, test_results]).sort_values(by='input_index').set_index('input_index')

# Save the results to a CSV file
results_file_path = '/tmp/prediction_results.csv'
results.to_csv(results_file_path)
print(f'Results saved locally at {results_file_path}')

# Set your AWS S3 bucket and results file name
bucket_name = 'nlpclaims'
s3_results_key = 'model_predictions/prediction_results.csv'  # S3 object key (path + filename)

# Upload results file to S3
upload_to_s3(results_file_path, bucket_name, s3_results_key)

# Save model weights to a temporary file
with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
    temp_file_path = temp_file.name
    model.save_weights(temp_file_path)
    print(f"Model weights saved locally at {temp_file_path}")

# Set your AWS S3 bucket and weights file name
s3_weights_key = 'model_predictions/model_weights_np.h5'  # S3 object key (path + filename)

# Upload weights file to S3
upload_to_s3(temp_file_path, bucket_name, s3_weights_key)

# Optionally save the entire model to the TensorFlow SavedModel format locally
save_entire_model = True  # Set to False if you do not need the entire model

if save_entire_model:
    local_model_dir = '/tmp/saved_model_np'
    model.save(local_model_dir, save_format='tf')
    print(f'Model saved locally at {local_model_dir}')

    # Verify the model directory exists
    if os.path.exists(local_model_dir):
        print(f"Model directory exists: {local_model_dir}")
    else:
        print("Model directory was not created successfully.")

    # Upload the model to S3
    s3_model_path = 'model_predictions/saved_model'
    for root, dirs, files in os.walk(local_model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_model_dir)
            s3_file_path = os.path.join(s3_model_path, relative_path)
            print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_file_path}")
            upload_to_s3(local_file_path, bucket_name, s3_file_path)


def list_s3_files(bucket, prefix):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(f"Found file: {obj['Key']}")
        else:
            print("No files found.")
    except Exception as e:
        print(f"Failed to list files in {bucket}/{prefix}. Error: {e}")

# List files in S3 bucket after uploading
list_s3_files(bucket_name, 'model_predictions/')

import tensorflow as tf
import boto3
import tempfile
import os
import pandas as pd



# Configure boto3 client with your credentials and region
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name='us-east-2')  # Update with your S3 bucket's region

# Function to upload file to S3
def upload_to_s3(local_file, bucket, s3_file):
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket)
        s3_client.upload_file(local_file, bucket, s3_file)
        print(f"File '{s3_file}' uploaded successfully to '{bucket}'")
    except boto3.exceptions.botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"The bucket '{bucket}' does not exist. Please create the bucket and try again.")
        else:
            print(f"Failed to upload {local_file} to {bucket}. Error: {e}")

# Function to list files in S3
def list_s3_files(bucket, prefix):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(f"Found file: {obj['Key']}")
        else:
            print("No files found.")
    except Exception as e:
        print(f"Failed to list files in {bucket}/{prefix}. Error: {e}")

# Convert tdp inputs to 2D tensors
X_train_tdp_input = tf.constant(X_train['tdp'], dtype=tf.float32)[:, tf.newaxis]
X_test_tdp_input = tf.constant(X_test['tdp'], dtype=tf.float32)[:, tf.newaxis]


# Make predictions
y_pred_train = model.predict([product_embeddings_train, categorical_input_train, X_train_tdp_input])
y_pred_test = model.predict([product_embeddings_test, categorical_input_test, X_test_tdp_input])

# Convert predictions to numpy arrays
y_pred_train = y_pred_train.flatten()
y_pred_test = y_pred_test.flatten()

# Create a DataFrame with the predictions
train_results = pd.DataFrame({
    'input_index': X_train.index,
    'true_values': y_train,
    'predicted_values': y_pred_train,
    'train_test_split': 'train'
})

test_results = pd.DataFrame({
    'input_index': X_test.index,
    'true_values': y_test,
    'predicted_values': y_pred_test,
    'train_test_split': 'test'
})

# Concatenate train and test results
results = pd.concat([train_results, test_results]).sort_values(by='input_index').set_index('input_index')

# Save the results to a CSV file
results_file_path = '/tmp/prediction_results.csv'
results.to_csv(results_file_path)
print(f'Results saved locally at {results_file_path}')

# Verify the results file exists
if os.path.exists(results_file_path):
    print(f"Results file exists: {results_file_path}")
else:
    print("Results file was not created successfully.")

# Set your AWS S3 bucket and results file name
bucket_name = 'claimsnlp'
s3_results_key = 'model_predictions/prediction_results.csv'  # S3 object key (path + filename)

# Upload results file to S3
upload_to_s3(results_file_path, bucket_name, s3_results_key)

# Save model weights to a temporary file
with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
    temp_file_path = temp_file.name
    model.save_weights(temp_file_path)
    print(f"Model weights saved locally at {temp_file_path}")

# Verify the weights file exists
if os.path.exists(temp_file_path):
    print(f"Weights file exists: {temp_file_path}")
else:
    print("Weights file was not created successfully.")

# Set your AWS S3 bucket and weights file name
s3_weights_key = 'model_predictions/model_weights_np.h5'  # S3 object key (path + filename)

# Upload weights file to S3
upload_to_s3(temp_file_path, bucket_name, s3_weights_key)

# Optionally save the entire model to the TensorFlow SavedModel format locally
save_entire_model = True  # Set to False if you do not need the entire model

if save_entire_model:
    local_model_dir = '/tmp/saved_model_np'
    model.save(local_model_dir, save_format='tf')
    print(f'Model saved locally at {local_model_dir}')

    # Verify the model directory exists
    if os.path.exists(local_model_dir):
        print(f"Model directory exists: {local_model_dir}")
    else:
        print("Model directory was not created successfully.")

    # Upload the model to S3
    s3_model_path = 'model_predictions/saved_model'
    for root, dirs, files in os.walk(local_model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_model_dir)
            s3_file_path = os.path.join(s3_model_path, relative_path)
            print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_file_path}")
            upload_to_s3(local_file_path, bucket_name, s3_file_path)

# List files in S3 bucket after uploading
list_s3_files(bucket_name, 'model_predictions/')
