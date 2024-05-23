from chalice import Chalice

app = Chalice(app_name='web-api')
dynamodb = boto3.resource('dynamodb')
dynamodb_table = dynamodb.Table(os.environ['DYNAMODB_TABLE_NAME'])

@app.route('/users', methods=['POST'])
def create_user():
    user = app.current_request.json_body
    dynamodb_table.put_item(Item=user)
    return user

@app.route('/user/{username}', methods=['GET'])
def get_user(username):
    response = dynamodb_table.get_item(Key={'username': username})
    return response['Item']

@app.route('/users/{username}', methods=['DELETE'])
def delete_user(username):
    dynamodb_table.delete_item(Key={'username': username})

# def index():
    # return {'hello': 'world'}


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
