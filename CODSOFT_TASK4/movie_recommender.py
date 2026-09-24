import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("movies.csv")

# Combine features
movies["combined_features"] = (
    movies["genre"].fillna("") + " " +
    movies["keywords"].fillna("") + " " +
    movies["overview"].fillna("")
)

# Convert text into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(movies["combined_features"])

# Calculate similarity
similarity = cosine_similarity(feature_matrix)


def recommend_movies(movie_title, number_of_recommendations=5):
    movie_title = movie_title.lower()

    matches = movies[movies["title"].str.lower() == movie_title]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    similarity_scores = list(enumerate(similarity[movie_index]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:number_of_recommendations + 1]:
        recommendations.append(movies.iloc[index]["title"])

    return recommendations


