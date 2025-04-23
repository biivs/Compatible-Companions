import json
import gc
import os
import re
import anthropic

client = anthropic.Anthropic(
    api_key="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"
)

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

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64

from uszipcode import SearchEngine
from math import radians, sin, cos, sqrt, atan2

def get_coords_from_zip(zip_code):
    search = SearchEngine(simple_zipcode=True)
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

    query_vec = tfidf_vectorizer.transform([query])
    tfidf_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    query_lsa = svd.transform(query_vec)
    lsa_sim = cosine_similarity(query_lsa, lsa_matrix).flatten()
    query_embedding = semantic_model.encode([query], convert_to_tensor=True)
    semantic_sim = cosine_similarity(query_embedding.cpu().numpy(), semantic_embeddings.cpu().numpy()).flatten()

    # Add individual scores to DataFrame
    animals_df['tfidf_score'] = tfidf_sim
    animals_df['lsa_score'] = lsa_sim
    animals_df['semantic_score'] = semantic_sim
    animals_df['score'] = 0.4 * semantic_sim + 0.3 * tfidf_sim + 0.3 * lsa_sim

    penalty = 0.3
    animals_df.loc[animals_df['full_description'].isnull(), 'score'] -= penalty
    animals_df['score'] = animals_df['score'].clip(lower=0)

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

    matches['image_url'] = matches['photos'].apply(
        lambda x: x[0]['small'] if isinstance(x, list) and x else "https://via.placeholder.com/300"
    )

    return matches[['id', 'name', 'url', 'type', 'species', 'age', 'gender', 'status',
                    'image_url', 'full_description', 'score', 'tfidf_score', 'lsa_score', 'semantic_score', 'distance']].to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/survey", methods=["GET", "POST"])
def survey():
    if request.method == "POST":
        answers = request.form.to_dict()

        prompt = "A user filled out a pet compatibility survey. Here are their responses:\n"
        for question, answer in answers.items():
            prompt += f"- {question.replace('_', ' ').capitalize()}: {answer.capitalize()}\n"
        prompt += "\nBased on this, recommend an ideal pet companion profile."

        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=400,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            message = response.content[0].text
        except Exception as e:
            message = f"Something went wrong: {str(e)}"

        return render_template("survey_result.html", message=message)

    return render_template("survey.html")

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
        return jsonify({"error": "Both id and query parameters are required"}), 400

    try:
        animal = animals_df[animals_df['id'] == int(animal_id)].iloc[0]
    except IndexError:
        return jsonify({"error": "Animal not found"}), 404

    all_traits = ['playful', 'calm', 'affectionate', 'energetic', 'friendly', 'gentle', 'independent']
    query = query.lower()
    traits_in_query = [trait for trait in all_traits if trait in query]

    if not traits_in_query:
        traits_in_query = ['playful', 'calm', 'friendly']  # default fallback traits

    animal_text = (animal['full_description'] or "").lower()

    def score_trait(trait):
        return 1.0 if re.search(rf"\b{re.escape(trait)}\b", animal_text) else 0.1  # crude scoring

    trait_scores = [score_trait(trait) for trait in traits_in_query]

    N = len(traits_in_query)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    trait_scores += trait_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, trait_scores, color='#FF7043', linewidth=2, marker='o')
    ax.fill(angles, trait_scores, color='#FF7043', alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), traits_in_query, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", fontsize=8)
    ax.grid(True, linestyle='--', color='gray', alpha=0.3)
    ax.set_title(f"Trait Match for: {animal['name']}", y=1.1, fontsize=14)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5001)
app.run(debug=True, host="0.0.0.0", port=5001)
