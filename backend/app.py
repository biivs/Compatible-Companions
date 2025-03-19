import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

animals_df = pd.DataFrame(data)
print(animals_df.head())

if 'id' not in animals_df.columns or 'full_description' not in animals_df.columns:
    raise ValueError("Expected 'id' and 'full_description' fields in JSON data")

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    query = query.lower()
    matches = animals_df[animals_df['full_description'].str.lower().str.contains(query, na=False)]
    matches['image_url'] = matches['photos'].apply(lambda x: x[0]['small'] if isinstance(x, list) and x else "https://via.placeholder.com/300")
    matches_filtered = matches[['id', 'name', 'url', 'type', 'species', 'age', 'gender', 'status', 'image_url','full_description']]
    return matches_filtered.to_json(orient='records')


@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/animals")
def animals_search():
    text = request.args.get("query")
    if not text:
        return jsonify({"error": "Query parameter is required"}), 400
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)