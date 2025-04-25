# from memory_profiler import profile
from dotenv import load_dotenv
load_dotenv()
import json
import gc
import os
import re
#import google.generativeai as genai
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64

from uszipcode import SearchEngine
from math import radians, sin, cos, sqrt, atan2

from spiderchart import (
    create_spider_chart,
    reverse_index,
    get_relevant_adjectives,
    score_description
    
)

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
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))

json_file_path = os.path.join(current_directory, 'data', 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)

with open(os.path.join(current_directory, 'data', 'FINAL_cat_breed_summaries.json')) as f:
    cat_breed_data = json.load(f)

with open(os.path.join(current_directory, 'data', 'FINAL_dog_breed_summaries.json')) as f:
    dog_breed_data = json.load(f)


animals_df = pd.DataFrame(data)

del data
gc.collect()

# Initialize TF-IDF Vectorizer and SentenceTransformer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(animals_df['full_description'].fillna(""))

# svd
n_components = 100  # Number of dimensions to reduce to
svd = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Precompute semantic embeddings for all descriptions
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
semantic_embeddings = semantic_model.encode(animals_df['full_description'].fillna(""), convert_to_tensor=True)

if 'id' not in animals_df.columns or 'full_description' not in animals_df.columns:
    raise ValueError("Expected 'id' and 'full_description' fields in JSON data")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default_secret_key")
CORS(app)

# def get_best_breed_match(prompt, type_):
#     model = genai.GenerativeModel("models/gemini-1.5-flash")
#     if type_.lower() == "dog":
#         breed_data = dog_breed_data
#     else:
#         breed_data = cat_breed_data

#     full_prompt = (
#         f"You are a {type_} breed expert.\n"
#         "Based on the following user preferences, choose ONE breed from this list that best matches.\n"
#         f"{json.dumps(breed_data, indent=2)}\n\n"
#         f"User responses:\n{prompt}\n\n"
#         "Respond with ONLY the breed name, no punctuation or extra text."
#     )

#     response = model.generate_content(full_prompt)
#     return response.text.strip()

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

def preprocess_query(query):
    query = query.lower()
    type_keywords = {"cat": 1, "dog": 1}
    detected_type = None

    for keyword, _ in type_keywords.items():
        if re.search(rf"\b{keyword}\b", query):
            detected_type = keyword
            break

    query = re.sub(r"\b(cat|dog)\b", "", query).strip()

    return query, detected_type

#@profile
def json_search(query, gender=None, age=None, animal_type=None, user_lat=None, user_lon=None):
    query = query.lower()
    query, detected_type = preprocess_query(query)
    if detected_type:
        animal_type = detected_type

    local_df = animals_df.copy()

    if gender:
        local_df = local_df[local_df['gender'].str.lower() == gender.lower()]
    if age:
        local_df = local_df[local_df['age'].str.lower() == age.lower()]
    if animal_type:
        valid_types = ["dog", "cat"]
        normalized_type = animal_type.strip().lower()
        if normalized_type in valid_types:
            local_df = local_df[local_df['type'].str.lower() == normalized_type]
        else:
            local_df = local_df[local_df['type'].str.lower().isin(valid_types)]

    local_df = local_df.reset_index(drop=True)
    local_df = local_df[local_df['full_description'].notnull()]

    # query_vec = tfidf_vectorizer.transform([query])
    # tfidf_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # query_lsa = svd.transform(query_vec)
    # lsa_sim = cosine_similarity(query_lsa, lsa_matrix).flatten()
    # query_embedding = semantic_model.encode([query], convert_to_tensor=True)
    # semantic_sim = cosine_similarity(query_embedding.cpu().numpy(), semantic_embeddings.cpu().numpy()).flatten()
    filtered_tfidf_matrix = tfidf_vectorizer.transform(local_df['full_description'].fillna(""))
    query_vec = tfidf_vectorizer.transform([query])
    tfidf_sim = cosine_similarity(query_vec, filtered_tfidf_matrix).flatten()

    filtered_lsa_matrix = svd.transform(filtered_tfidf_matrix)
    query_lsa = svd.transform(query_vec)
    lsa_sim = cosine_similarity(query_lsa, filtered_lsa_matrix).flatten()

    filtered_descriptions = local_df['full_description'].fillna("").reset_index(drop=True)
    filtered_semantic_embeddings = semantic_model.encode(filtered_descriptions, convert_to_tensor=True)
    query_embedding = semantic_model.encode([query], convert_to_tensor=True)
    semantic_sim = cosine_similarity(query_embedding.cpu().numpy(), filtered_semantic_embeddings.cpu().numpy()).flatten()


    local_df['tfidf_score'] = tfidf_sim
    local_df['lsa_score'] = lsa_sim
    local_df['semantic_score'] = semantic_sim
    local_df['score'] = 0.4 * semantic_sim + 0.3 * tfidf_sim + 0.3 * lsa_sim

    # penalty = 0.3
    # local_df.loc[local_df['full_description'].isnull(), 'score'] -= penalty
    # local_df['score'] = local_df['score'].clip(lower=0)
    if detected_type:
        type_penalty = 0.5
        type_match = local_df['type'].str.lower() == detected_type
        local_df['score'] -= (~type_match) * type_penalty
        local_df['score'] = local_df['score'].clip(lower=0)

    matches = local_df[local_df['score'] > 0.1].copy()

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

    matches['image_url'] = matches['photos'].apply(
        lambda x: x[0]['small'] if isinstance(x, list) and x else "https://via.placeholder.com/300"
    )

    # return matches[['id', 'name', 'url', 'type', 'species', 'age', 'gender', 'status', 'image_url', 'full_description', 'score', 'tfidf_score', 'lsa_score', 'semantic_score', 'distance']].to_json(orient='records')
   
    #TENTATIVE FIX BELOW

    cols_to_return = ['id', 'name', 'url', 'type', 'species', 'age', 'gender', 'status',
                  'image_url', 'full_description', 'score']

    # Only add optional scoring columns if they exist 
    for col in ['tfidf_score', 'lsa_score', 'semantic_score', 'distance']:
        if col in matches.columns:
            cols_to_return.append(col)
    
    return matches[cols_to_return].to_json(orient='records')

@app.route("/")
def home():
    recommended_breed = session.pop("recommended_breed", None)
    return render_template('base.html', recommended_breed=recommended_breed)

@app.route("/survey", methods=["GET", "POST"])
def survey():
    if request.method == "POST":
        answers = request.form.to_dict()
        animal_type = answers.get("animal_type")

        if animal_type == "dog":
            dog_key = ''.join(answers.get(f"dog_q{i+1}", "N").upper() for i in range(6))
            with open("data/dog_breed_mappings.json") as f:
                dog_breed_map = json.load(f)
            breed = dog_breed_map.get(dog_key, "Unknown")
            summary = next(
                (b["Trait Summary"] for b in dog_breed_data if b["Breed"].lower() == breed.lower()),
                "Sorry, no summary found."
            )
            session["recommended_breed"] = breed
            return render_template("survey_result.html", breed=breed, summary=summary)

        elif animal_type == "cat":
            cat_key = ''.join(answers.get(f"cat_q{i+1}", "N").upper() for i in range(4))
            with open("data/cat_breed_mappings.json") as f:
                cat_breed_map = json.load(f)
            breed = cat_breed_map.get(cat_key, "Unknown")
            summary = next(
                (b["Trait Summary"] for b in cat_breed_data if b["Breed"].lower() == breed.lower()),
                "Sorry, no summary found."
            )
            session["recommended_breed"] = breed
            return render_template("survey_result.html", breed=breed, summary=summary)

        else:
            return "Invalid animal type", 400

    return render_template("survey.html")



# @app.route("/survey", methods=["GET", "POST"])
# def survey():
#     if request.method == "POST":
#         answers = request.form.to_dict()

#         prompt = (
#             "A user filled out a pet compatibility survey. Here are their responses:\n"
#             f"- Interested in: {answers.get('animal_type', 'not specified').capitalize()}\n"
#             f"- Time commitment: {answers.get('time_commitment', 'not specified').capitalize()}\n"
#             f"- Other pets: {answers.get('other_pets', 'not specified').capitalize()}\n"
#             f"- Size: {answers.get('size', 'not specified').capitalize()}\n"
#             f"- Allergies: {answers.get('allergies', 'not specified').capitalize()}\n"
#             f"- Children: {answers.get('children', 'not specified').capitalize()}\n"
#             f"- Okay with senior pets: {answers.get('senior_ok', 'not specified').capitalize()}\n"
#             f"- Outdoor space: {answers.get('outdoor_space', 'not specified').capitalize()}\n"

#         )

#         type_ = answers.get("animal_type", "cat")

#         try:
#             breed = get_best_breed_match(prompt, type_)
#             session["recommended_breed"] = breed
#             message = f"Based on your survey, we recommend the {type_} breed: {breed}"
#             summary = next((item["Trait Summary"] for item in (dog_breed_data if type_ == "dog" else cat_breed_data) if item["Breed"].lower() == breed.lower()), "No summary found.")
#             reason = None
#         except Exception as e:
#             message = f"Something went wrong: {str(e)}"
#             breed = None
#             reason = None

#         return render_template("survey_result.html", message=message, breed=breed, reason=reason, summary=summary)

#     return render_template("survey.html")

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
    zip_code = request.args.get("zip_code")
    user_lat, user_lon = get_coords_from_zip(zip_code) if zip_code else (None, None)
    start = int(request.args.get("start", 0))
    limit = int(request.args.get("limit", 20))

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results_json = json.loads(json_search(query, gender, age, type_, user_lat, user_lon))
    paginated = results_json[start:start + limit]
    return jsonify({
        "results": paginated,
        "total": len(results_json)
    })


@app.route("/similarity_chart")
def similarity_chart():
    animal_id = request.args.get("id")
    query = request.args.get("query", "")

    if not animal_id or not query:
        return jsonify({"error": "Both 'id' and 'query' parameters are required"}), 400

    try:
        animal_id = int(animal_id)
    except ValueError:
        return jsonify({"error": "Invalid 'id' parameter"}), 400

    try:
        animal = animals_df[animals_df['id'] == animal_id].iloc[0]
        description = animal['full_description']
    except IndexError:
        return jsonify({"error": "Animal not found"}), 404

    adjectives = get_relevant_adjectives(query, description)
    
    query_scores = score_description(query, adjectives)
    description_scores = score_description(description, adjectives)

    buf = create_spider_chart(
        adjectives,
        query_scores,
        description_scores,
        f"Trait Comparison: Your Preferences vs {animal['name']}'s Traits"
    )

    if not buf:
        return jsonify({"error": "Failed to generate chart"}), 500

    return send_file(buf, mimetype='image/png')

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)