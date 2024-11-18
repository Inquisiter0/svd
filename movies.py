from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import kagglehub

# Initialize Flask app
app = Flask(__name__, template_folder="movies_templates",static_folder="movies_templates/static")

# Download dataset
#path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
#print("Path to dataset files:", path)

movies_path = '/home/ombir/svd_model/datasets/movies.csv'
ratings_path = '/home/ombir/svd_model/datasets/ratings.csv'

# Load datasets
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
#ratings = ratings.head(100000)  # Adjust for memory usage

# Create user-item matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
sparse_matrix = csr_matrix(user_movie_matrix.values)

# SVD-based recommendation function
svd = TruncatedSVD(n_components=50, random_state=42)
matrix = svd.fit_transform(sparse_matrix)

def svd_recommend(user_id, n_recommendations=10):
    user_idx = user_id - 1
    similarity = cosine_similarity([matrix[user_idx]], matrix)[0]
    similar_users = similarity.argsort()[-n_recommendations:]
    recommended_movies = user_movie_matrix.columns[similar_users].tolist()
    return movies[movies['movieId'].isin(recommended_movies)]['title'].tolist()

# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    user_id = None

    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
            recommendations = svd_recommend(user_id)
        except ValueError:
            recommendations = "Invalid User ID"

    return render_template('sml.html', user_id=user_id, recommendations=recommendations)





if __name__ == "__main__":
    app.run(debug=True)
