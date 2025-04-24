import matplotlib.pyplot as plt
import numpy as np
import re
import os
import json
import io
from collections import defaultdict

current_directory = os.path.dirname(os.path.abspath(__file__))

synonyms_file_path = os.path.join(current_directory, 'data', 'synonyms.json')
with open(synonyms_file_path, 'r') as f:
    synonyms_dict = json.load(f)

def build_reverse_index(synonyms_dict):
    reverse_index = {}
    for main_adjective, synonyms in synonyms_dict.items():
        for synonym in synonyms:
            reverse_index[synonym] = main_adjective
        reverse_index[main_adjective] = main_adjective
    return reverse_index

reverse_index = build_reverse_index(synonyms_dict)

adjectives_ranked_file_path = os.path.join(current_directory, 'data', 'adjectives_ranked.txt')
def load_top_adjectives(filename, top_n=100):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    top_adjectives = []
    for line in lines[:top_n]:
        parts = line.split(':')
        if parts:
            top_adjectives.append(parts[0].strip())
    return top_adjectives

top_adjectives = load_top_adjectives(adjectives_ranked_file_path, top_n=100)

def get_relevant_adjectives(query, description, top_n=10):
    # Terms to exclude from the chart
    EXCLUDED_TERMS = {'dog', 'cat', 'puppy', 'kitten', 'pet', 'animal'}
    
    query_tokens = set(re.findall(r'\b\w+\b', query.lower())) - EXCLUDED_TERMS
    
    mandatory_adjectives = {reverse_index.get(token, token) for token in query_tokens}
    
    desc_tokens = set(re.findall(r'\b\w+\b', description.lower())) - EXCLUDED_TERMS
    desc_adjs = {reverse_index.get(token, token) for token in desc_tokens}
    
    mandatory_adjectives -= EXCLUDED_TERMS
    desc_adjs -= EXCLUDED_TERMS
    
    query_synonyms = set()
    for adj in mandatory_adjectives:
        query_synonyms.add(adj)
        if adj in synonyms_dict:
            query_synonyms.update(synonyms_dict[adj])
    
    adjective_scores = {}
    
    for adj in top_adjectives:
        if adj in EXCLUDED_TERMS:
            continue
            
        score = 0
        if adj in mandatory_adjectives:
            score += 3
        elif adj in query_synonyms:
            score += 2
        if adj in desc_adjs:
            score += 1
        
        if score > 0:
            adjective_scores[adj] = score
    
    selected = list(mandatory_adjectives)
    
    sorted_adjs = sorted(
        [adj for adj in adjective_scores.keys() if adj not in mandatory_adjectives],
        key=lambda x: (-adjective_scores[x], top_adjectives.index(x) if x in top_adjectives else float('inf'))
    )
    
    remaining_slots = top_n - len(selected)
    selected.extend(sorted_adjs[:remaining_slots])
    
    if len(selected) < top_n:
        remaining = top_n - len(selected)
        additional = [adj for adj in top_adjectives 
                    if adj not in selected and adj not in EXCLUDED_TERMS][:remaining]
        selected.extend(additional)
    
    return selected[:top_n]

def score_description(text, adjectives):
    """Returns scores between 0-1 for each adjective based on frequency"""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    
    scores = []
    for adj in adjectives:
        exact_matches = sum(1 for token in tokens if token == adj)
        
        synonym_matches = 0
        if adj in synonyms_dict:
            synonyms = synonyms_dict[adj]
            synonym_matches = sum(1 for token in tokens if token in synonyms)
        
        total_matches = exact_matches + synonym_matches
        score = min(1.0, 0.3 + total_matches * 0.35) 
        scores.append(score)
    
    return scores

def create_spider_chart(adjectives, query_scores, description_scores, title):
    N = len(adjectives)
    if N == 0:
        return None
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], adjectives, color='grey', size=10, ha='center')
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.1)

    values_query = query_scores + query_scores[:1]
    values_description = description_scores + description_scores[:1]

    ax.plot(angles, values_query, linewidth=2, linestyle='solid', label='Your Preferences', color='#1f77b4')
    ax.fill(angles, values_query, '#1f77b4', alpha=0.2)

    ax.plot(angles, values_description, linewidth=2, linestyle='solid', label="Pet's Traits", color='#ff7f0e')
    ax.fill(angles, values_description, '#ff7f0e', alpha=0.2)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title(title, size=14, color="black", y=1.15)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)

    return buf