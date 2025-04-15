# Refactored app.py with reduced global memory usage

import json
import gc
import os
import re
from flask import Flask, render_template, request, jsonify, send_file
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
from uszipcode import SearchEngine
from math import radians, sin, cos, sqrt, atan2
import io

app = Flask(__name__)
CORS(app)

# Lazy-load models and data
_tf_cache = {}
_model_cache = {}


def get_coords_from_zip(zip_code):
    search = SearchEngine() 
    result = search.by_zipcode(zip_code)
    return (result.lat, result.lng) if result else (None, None)


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_animals():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init.json')
    with open(path, 'r') as f:
        animals = pd.DataFrame(json.load(f))
    return animals


def load_models(animals_df):
    if not _model_cache:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(animals_df['full_description'].fillna(""))

        svd = TruncatedSVD(n_components=100, random_state=42)
        lsa_matrix = svd.fit_transform(tfidf_matrix)

        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        sem_embeddings = semantic_model.encode(animals_df['full_description'].fillna(""), convert_to_tensor=True)

        _model_cache['tfidf'] = tfidf
        _model_cache['tfidf_matrix'] = tfidf_matrix
        _model_cache['svd'] = svd
        _model_cache['lsa_matrix'] = lsa_matrix
        _model_cache['semantic_model'] = semantic_model
        _model_cache['semantic_embeddings'] = sem_embeddings

    return _model_cache


def json_search(query, gender=None, age=None, animal_type=None, user_lat=None, user_lon=None):
    query = query.lower()
    animals_df = load_animals()
    models = load_models(animals_df)

    tfidf_vec = models['tfidf'].transform([query])
    tfidf_sim = cosine_similarity(tfidf_vec, models['tfidf_matrix']).flatten()

    lsa_sim = cosine_similarity(models['svd'].transform(tfidf_vec), models['lsa_matrix']).flatten()

    query_embedding = models['semantic_model'].encode([query], convert_to_tensor=True)
    semantic_sim = cosine_similarity(query_embedding.cpu().numpy(), models['semantic_embeddings'].cpu().numpy()).flatten()

    animals_df['score'] = 0.4 * semantic_sim + 0.3 * tfidf_sim + 0.3 * lsa_sim
    animals_df['score'] = animals_df['score'].where(animals_df['full_description'].notnull(), animals_df['score'] - 0.3).clip(lower=0)

    matches = animals_df[animals_df['score'] > 0.1].copy()
    if gender: matches = matches[matches['gender'].str.lower() == gender.lower()]
    if age: matches = matches[matches['age'].str.lower() == age.lower()]
    if animal_type: matches = matches[matches['type'].str.lower() == animal_type.lower()]

    if user_lat and user_lon:
        user_lat, user_lon = float(user_lat), float(user_lon)

        def calc_dist(c):
            zip_code = c.get('address', {}).get('postcode') if isinstance(c, dict) else None
            if not zip_code: return float('inf')
            lat2, lon2 = get_coords_from_zip(zip_code)
            return haversine(user_lat, user_lon, lat2, lon2) if lat2 and lon2 else float('inf')

        matches['distance'] = matches['contact'].apply(calc_dist)
        matches = matches[matches['distance'] < 100].sort_values('distance')
    else:
        matches['distance'] = None
        matches = matches.sort_values('score', ascending=False)

    matches['image_url'] = matches['photos'].apply(lambda x: x[0]['small'] if isinstance(x, list) and x else "https://via.placeholder.com/300")
    return matches[['id', 'name', 'url', 'type', 'species', 'age', 'gender', 'status',
                    'image_url', 'full_description', 'score', 'distance']].to_json(orient='records')


@app.route("/")
def home():
    return render_template('base.html', title="Compatible Companions")


@app.route("/animals")
def animals_search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    return json_search(
        query,
        gender=request.args.get("gender"),
        age=request.args.get("age"),
        animal_type=request.args.get("type"),
        user_lat=request.args.get("user_lat"),
        user_lon=request.args.get("user_lon")
    )


@app.route("/similarity_chart")
def similarity_chart():
    animal_id = request.args.get("id")
    query = request.args.get("query", "")
    if not animal_id or not query:
        return jsonify({"error": "Both id and query parameters are required"}), 400

    animals_df = load_animals()
    try:
        animal = animals_df[animals_df['id'] == int(animal_id)].iloc[0]
    except IndexError:
        return jsonify({"error": "Animal not found"}), 404

    all_traits = ['playful', 'calm', 'affectionate', 'energetic', 'friendly', 'gentle', 'independent']
    traits_in_query = [t for t in all_traits if t in query.lower()] or ['playful', 'calm', 'friendly']
    animal_text = (animal['full_description'] or "").lower()

    scores = [1.0 if re.search(rf"\\b{re.escape(trait)}\\b", animal_text) else 0.1 for trait in traits_in_query]
    angles = np.linspace(0, 2 * np.pi, len(traits_in_query), endpoint=False).tolist() + [0]
    scores += scores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, color='#FF7043', linewidth=2, marker='o')
    ax.fill(angles, scores, color='#FF7043', alpha=0.25)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), traits_in_query)
    ax.set_ylim(0, 1)
    ax.set_title(f"Trait Match for: {animal['name']}", y=1.1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5001)
