import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess_data(data):
    """Preprocess the dataset by combining genres, tags, and keywords."""
    data['combined_features'] = data['genres'] + " " + data['tags'] + " " + data['keywords']
    data['combined_features'] = data['combined_features'].fillna("")
    return data

def build_similarity_matrix(data):
    """Build a cosine similarity matrix from the combined features."""
    vectorizer = CountVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(data['combined_features'])
    return cosine_similarity(feature_matrix)

def recommend_movies(title, similarity_matrix, indices, data, filters=None, num_recommendations=10):
    """Recommend movies based on a title with optional filters."""
    if title not in indices:
        return f"Movie '{title}' not found in the dataset."

    # Get the index of the movie in the dataset
    idx = indices[title]

    # Calculate similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # Exclude the movie itself

    # Apply filters if specified
    if filters:
        sim_scores = [
            (i, score) for i, score in sim_scores
            if all(data.loc[i, key] == value for key, value in filters.items())
        ]

    # Get the indices of the recommended movies
    movie_indices = [i[0] for i in sim_scores[:num_recommendations]]
    return data['title'].iloc[movie_indices].tolist()

def get_popular_movies(data, top_n=10):
    """Retrieve top popular movies based on vote average and popularity."""
    popular_movies = data.sort_values(['vote_average', 'popularity'], ascending=False)
    return popular_movies['title'].head(top_n).tolist()

# Load the dataset
file_path = 'clean-data.csv'
data = pd.read_csv(file_path)



# Preprocess the dataset
data = preprocess_data(data)

# Build the similarity matrix
similarity_matrix = build_similarity_matrix(data)

# Create a Series to map movie titles to indices
movie_indices = pd.Series(data.index, index=data['title']).drop_duplicates()

# CLI Interface for recommendations
print("Welcome to the Movie Recommendation System!")
while True:
    print("\nOptions:")
    print("1. Get recommendations based on a movie title")
    print("2. See a list of popular movies")
    print("3. Exit")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        movie_title = input("Enter the movie title: ").strip()
        
        # Optional filters
        filter_by_genre = input("Filter by genre (leave blank for no filter): ").strip()
        filter_by_year = input("Filter by release year (leave blank for no filter): ").strip()
        
        filters = {}
        if filter_by_genre:
            filters['genres'] = filter_by_genre.lower()
        if filter_by_year:
            filters['year'] = filter_by_year

        # Get recommendations
        recommendations = recommend_movies(movie_title, similarity_matrix, movie_indices, data, filters)
        print("\nRecommended Movies:")
        print(recommendations if isinstance(recommendations, list) else recommendations)

    elif choice == "2":
        print("\nPopular Movies:")
        popular_movies = get_popular_movies(data)
        print(popular_movies)

    elif choice == "3":
        print("Thank you for using the Movie Recommendation System! Goodbye!")
        break

    else:
        print("Invalid choice. Please try again.")
