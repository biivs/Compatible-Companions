import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

from uszipcode import SearchEngine
from math import radians, sin, cos, sqrt, atan2

def get_coords_from_zip(zip_code):
    search = SearchEngine() 
    result = search.by_zipcode(zip_code)
    if result:
        return result.lat, result.lng
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

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

# Initialize TF-IDF Vectorizer and SentenceTransformer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(animals_df['full_description'].fillna(""))

# Precompute semantic embeddings for all descriptions
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
semantic_embeddings = semantic_model.encode(animals_df['full_description'].fillna(""), convert_to_tensor=True)

if 'id' not in animals_df.columns or 'full_description' not in animals_df.columns:
    raise ValueError("Expected 'id' and 'full_description' fields in JSON data")

app = Flask(__name__)
CORS(app)

# # Sample search using json with pandas
# def json_search(query):
#     query = query.lower()
    
#     # --- TF-IDF Similarity ---
#     query_vec = tfidf_vectorizer.transform([query])
#     tfidf_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

#     # --- Semantic Similarity ---
#     query_embedding = semantic_model.encode([query], convert_to_tensor=True)
#     semantic_sim = cosine_similarity(query_embedding.cpu().numpy(), semantic_embeddings.cpu().numpy()).flatten()

#     # --- Combine Scores ---
#     combined_score = 0.5 * tfidf_sim + 0.5 * semantic_sim
#     animals_df['score'] = combined_score

#     # --- Filter and sort ---
#     matches = animals_df[combined_score > 0.1].copy()
#     matches = matches.sort_values(by='score', ascending=False)

#     # --- Image placeholder fallback ---
#     matches['image_url'] = matches['photos'].apply(
#         lambda x: x[0]['small'] if isinstance(x, list) and x else "https://via.placeholder.com/300"
#     )

#     matches_filtered = matches[['id', 'name', 'url', 'type', 'species', 'age', 'gender', 'status', 'image_url', 'full_description', 'score']]
#     return matches_filtered.to_json(orient='records')

def json_search(query, gender=None, age=None, animal_type=None, user_lat=None, user_lon=None):
    query = query.lower()

    # Similarity Scores
    query_vec = tfidf_vectorizer.transform([query])
    tfidf_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    query_embedding = semantic_model.encode([query], convert_to_tensor=True)
    semantic_sim = cosine_similarity(query_embedding.cpu().numpy(), semantic_embeddings.cpu().numpy()).flatten()

    combined_score = 0.5 * semantic_sim + 0.5 * tfidf_sim
    animals_df['score'] = combined_score

    #penalty for empty description
    penalty = 0.3
    animals_df.loc[animals_df['full_description'].isnull(), 'score'] -= penalty
    animals_df['score'] = animals_df['score'].clip(lower=0)  # Avoid negative scores

    # Filter and sort
    matches = animals_df[animals_df['score'] > 0.1].copy()

    if gender:
        matches = matches[matches['gender'].str.lower() == gender.lower()]
    if age:
        matches = matches[matches['age'].str.lower() == age.lower()]
    if animal_type:
        matches = matches[matches['type'].str.lower() == animal_type.lower()]

    if user_lat and user_lon:
        user_lat = float(user_lat)
        user_lon = float(user_lon)

        def calc_distance(contact):
            zip_code = contact.get('address', {}).get('postcode') if contact else None
            if not zip_code:
                return float('inf')
            lat2, lon2 = get_coords_from_zip(zip_code)
            if lat2 is None or lon2 is None:
                return float('inf')
            return haversine(user_lat, user_lon, lat2, lon2)

        matches['distance'] = matches['contact'].apply(calc_distance)
        matches = matches[matches['distance'] < 100]
        matches = matches.sort_values(by='distance')
    else:
        matches['distance'] = None
        matches = matches.sort_values(by='score', ascending=False)

    # Image fallback
    matches['image_url'] = matches['photos'].apply(
        lambda x: x[0]['small'] if isinstance(x, list) and x else "https://via.placeholder.com/300"
    )

    return matches[['id', 'name', 'url', 'type', 'species', 'age', 'gender', 'status',
                    'image_url', 'full_description', 'score', 'distance']].to_json(orient='records')
@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

# @app.route("/animals")
# def animals_search():
#     text = request.args.get("query")
#     if not text:
#         return jsonify({"error": "Query parameter is required"}), 400
#     return json_search(text)

@app.route("/animals")
def animals_search():
    query = request.args.get("query", "")
    gender = request.args.get("gender")
    age = request.args.get("age")
    type_ = request.args.get("type", "")
    user_lat = request.args.get("user_lat")
    user_lon = request.args.get("user_lon")

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    return json_search(query, gender, age, type_,user_lat, user_lon)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)