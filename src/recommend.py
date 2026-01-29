import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, preprocess_text
from src.model import build_vectorizer, compute_similarities

def recommend_talks(query, df, vectorizer, tfidf_matrix, top_n=5):
    cos_sim, pea_sim = compute_similarities(query, df, vectorizer, tfidf_matrix)

    df = df.copy()
    df['cos_sim'] = cos_sim
    df['pea_sim'] = pea_sim

    df = df.sort_values(
        by=['cos_sim', 'pea_sim'],
        ascending=False
    )

    return df[['main_speaker', 'details']].head(top_n)

if __name__ == "__main__":
    df = load_data("data/tedx_dataset.csv")
    df = preprocess_text(df)

    vectorizer, tfidf_matrix = build_vectorizer(df['details'])

    query = [
        "Climate change and impact on health and carbon footprint"
    ]

    recommendations = recommend_talks(
        query, df, vectorizer, tfidf_matrix
    )

    print("\nRecommended TED Talks:\n")
    print(recommendations.to_string(index=False))
