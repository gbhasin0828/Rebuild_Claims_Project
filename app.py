import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout

app = Flask(__name__)

# Define the SequentialModel class
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

    def get_config(self):
        config = super(SequentialModel, self).get_config()
        config.update({"output_dim": self.output_layer.units})
        return config

    @classmethod
    def from_config(cls, config):
        output_dim = config.pop("output_dim", 1)
        return cls(output_dim=output_dim)

# Declare global variables for the model, bert_model, and tokenizer
model = None
bert_model = None
tokenizer = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tdp = 6140  # Hardcoded value for TDP
    upcs = float(data['UPCs'])
    category = data['Category']
    manufacturer = data['Manufacturer']
    products = data['Products']
    time = data['Time']
    sub_category = data['SubCategory']
    segment = data['Segment']

    # Tokenize the string inputs using BERT tokenizer
    def tokenize_inputs(texts):
        tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}

    tensor_products = tokenize_inputs([products])
    tensor_category = tokenize_inputs([category])
    tensor_manufacturer = tokenize_inputs([manufacturer])
    tensor_time = tokenize_inputs([time])
    tensor_sub_category = tokenize_inputs([sub_category])
    tensor_segment = tokenize_inputs([segment])

    bert_output_product = bert_model(tensor_products)[1]
    bert_output_category = bert_model(tensor_category)[1]
    bert_output_manufacturer = bert_model(tensor_manufacturer)[1]
    bert_output_time = bert_model(tensor_time)[1]
    bert_output_sub_category = bert_model(tensor_sub_category)[1]
    bert_output_segment = bert_model(tensor_segment)[1]

    concatenate_string_inputs = tf.concat([bert_output_product, bert_output_category, bert_output_manufacturer, bert_output_time, bert_output_sub_category, bert_output_segment], axis=-1)

    # Convert numerical inputs to tensors
    tensor_tdp = tf.convert_to_tensor([tdp], dtype=tf.float32)
    tensor_upcs = tf.convert_to_tensor([upcs], dtype=tf.float32)
    tensor_tdp = tf.reshape(tensor_tdp, (tensor_tdp.shape[0], 1))
    tensor_upcs = tf.reshape(tensor_upcs, (tensor_upcs.shape[0], 1))

    # Predict the sales
    prediction_tensor = model.predict([concatenate_string_inputs, tensor_tdp, tensor_upcs])
    predictions_reshaped_tensor = tf.reshape(prediction_tensor, (prediction_tensor.shape[0],))
    predictions_scaled_values = predictions_reshaped_tensor.numpy()

    # Note: Ensure min_max_scaler is loaded and used correctly if used for inverse transformation
    # original_sales_values = min_max_scaler.inverse_transform(predictions_scaled_values.reshape(-1, 1))

    # Return all prediction values
    return jsonify({
        'prediction_tensor': prediction_tensor.tolist(),
        'predictions_reshaped_tensor': predictions_reshaped_tensor.tolist(),
        'predictions_scaled_values': predictions_scaled_values.tolist(),
        # 'original_sales_values': original_sales_values.flatten().tolist()
    })

if __name__ == '__main__':
    # Initialize the model, bert_model, and tokenizer globally
    model = SequentialModel(1)
    # Build the model with the correct input shapes
    model([tf.zeros((1, 768)), tf.zeros((1, 13)), tf.zeros((1, 1))])
    # Load the weights
    model.load_weights('my_model/model_weights_np.h5')
    # Load BERT model and tokenizer
    bert_model = TFBertModel.from_pretrained('my_model')
    tokenizer = BertTokenizer.from_pretrained('my_model')
    # Run the app
    app.run(host='0.0.0.0', port=8080)
