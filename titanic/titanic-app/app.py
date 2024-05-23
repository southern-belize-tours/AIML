from chalice import Chalice
import boto3
import io
import logging
import pathlib
import sklearn

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = Chalice(app_name='titanic-app')
s3_client = boto3.client('s3')

app.api.cors = True
app.api_gateway_stage = 'api'  # Disable authentication
# app.vendored = '/vendor'

# Necessary libraries for machine learning
# import joblib
import pickle
import json
# import numpy as np

def load_model():
    # model_path = 's3://titanic-cali/model1.pkl'
    # model_path = pathlib.Path(__file__).parent / 'model1.pkl'
    model_data = s3_client.get_object(Bucket="titanic-cali", Key="model2.pkl")['Body'].read()
    # model_data_stream = io.BytesIO(model_path)
    # model = pickle.load(model_data_stream)
    logger.info(f"model data loaded from bucket")
    model = pickle.loads(model_data)

    # model = joblib.load(model_data_stream)
    logger.info(f"model path during load model: {model}")

    return model

def format_input(model_input):
    assert model_input, "Empty model input"
    # data = np.array(model_input).reshape(1,-1)
    data = []
    data.append(model_input)
    # data = model_input.reshape(1, -1)
    # assert len(data.shape) in (1,2), "Data should be either 1D or 2D"
    # return data
    return data

def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions

def format_response(rtype, k, v):
    return {"type": rtype, k: v}

def format_prediction(pred):
    return format_response("success", "prediction", pred)

def format_error(err):
    return format_response("error", "message", err)

# @app.iam_policy(name="s3ReadPolicy")
# def s3_read_policy():
#     return {
#         "Version": "2012-10-17",
#         "Statement": [
#             {
#                 "Effect": "Allow",
#                 "Action": [
#                     "s3:GetObject",
#                     "s3:ListBucket"
#                 ],
#                 "Resource": [
#                     "arn:aws:s3:::titanic-cali",
#                     "arn:aws:s3:::titanic-cali/*"
#                 ]
#             }
#         ]
#     }

@app.route('/')
def index():
    return {'hello': 'world'}

@app.route('/hello', methods=['POST', 'GET'])
def hello():
    return {'hello from hello': 'world from hello'}

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        input_data = app.current_request.json_body
        logger.info(f"Received input data: {input_data}")
        model = load_model()
        logger.info(f"Got the model successfully {model}")
        # Make predictions
        data = input_data['data']
        logger.info(f"Data before formatting: {data}")
        data = json.loads(data)
        logger.info(f"Data after json load: {data}")
        data = format_input(data)
        logger.info(f"Reshaped the input data successfully {data}")

        predictions = make_predictions(model, data)
        logger.info(f"Made predictions successfully {predictions.tolist()}")
        # return({"success": "very nice"})
        # return format_prediction(predictions[0])
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Error in predict function: {str(e)}")
        return format_error("Error making predictions")

    # input_data = app.curent_request.json_body
    return {"predicting": "got input data"}
    # try:
    #     # Load the model
    #     model = load_model()
    #     return {"success": "loaded model"}
    # except (e):
    #     return {"failure to load model": e}
    

    # Make predictions
    # predictions = make_predictions(model, input_data)

    # Return predictions as a JSON response
    # response = {
    #     # 'predictions': predictions.tolist()
    #     'you are predicting': 'predicting things'
    # }

    # return json.dumps(response)



# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#
