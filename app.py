import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from googleapiclient.discovery import build
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(page_title='STREAMFLIX', page_icon="🎬", layout='wide')

# Load data and model
@st.cache_data
def load_data_and_model():
    try:
        collab_df = pd.read_csv('modelling_data/collab_movies.csv')
        content_df = pd.read_csv('modelling_data/content_movies.csv')
        merged_df = pd.merge(collab_df, content_df, on='movieId').drop_duplicates(subset=['movieId'])
        with open('pickle_files/collaborative_model1.pkl', 'rb') as f:
            collab_model = pickle.load(f)
        return merged_df, collab_model
    except Exception as e:
        st.error(f"Error loading data and model: {e}")
        return None, None

merged_df, collab_model = load_data_and_model()

class CollabBasedModel:
    def __init__(self, collab_df, model):
        self.df = collab_df
        self.model = model

    def get_recommendations(self, user_ratings, n=5):
        new_user_id = self.df['user_id'].max() + 1
        # Create a DataFrame for new user ratings
        new_ratings_df = pd.DataFrame(user_ratings, columns=['movieId', 'rating'])
        new_ratings_df['user_id'] = new_user_id

        # Append new user ratings to the dataset
        self.df = pd.concat([self.df, new_ratings_df[['user_id', 'movieId', 'rating']]], ignore_index=True)

        # Train the model with the updated dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.df[['user_id', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)

        # Get recommendations
        movies_to_predict = self.df[~self.df['movieId'].isin([x[0] for x in user_ratings])]['movieId'].unique()
        predictions = [(movie_id, self.model.predict(new_user_id, movie_id).est) for movie_id in movies_to_predict]
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
        return recommendations[:n]

# Fetch movie poster
@st.cache_data
def fetch_poster(movie_title):
    try:
        tmdb_api_key = os.getenv('TMDB_API_KEY')
        url = f"https://api.themoviedb.org/3/search/movie?api_key={tmdb_api_key}&query={movie_title}"
        response = requests.get(url)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path', '')
            return "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else "https://via.placeholder.com/500x750.png?text=No+Poster+Available"
        else:
            return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"
    except Exception as e:
        st.warning(f"Error fetching poster for movie {movie_title}: {e}")
        return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"

# Get trailer URL
@st.cache_data
def get_trailer_url(movie_title):
    try:
        youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        
        # Search for the movie trailer
        search_response = youtube.search().list(
            q=f"{movie_title} official trailer",
            type='video',
            part='id,snippet',
            maxResults=1
        ).execute()
        
        # Get the first search result
        if search_response['items']:
            video_id = search_response['items'][0]['id']['videoId']
            return f"https://www.youtube.com/watch?v={video_id}"
        else:
            return None
    except Exception as e:
        st.warning(f"Error fetching trailer for {movie_title}: {e}")
        return None

# Main app
def main():
    st.title("🎬 Streamflix Movie Recommendation System")

    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home', 'Get Recommendations', 'Search Movies', 'About'])

    if page == 'Home':
        st.header('🔥 Top Trending Movies')
        top_movies = merged_df['movieId'].value_counts().head(20).index
        for movie_id in top_movies:
            movie = merged_df[merged_df['movieId'] == movie_id].iloc[0]
            col1, col2 = st.columns([1, 3])
            with col1:
                poster_url = fetch_poster(movie['title'])
                st.image(poster_url, width=150)
            with col2:
                st.subheader(movie['title'])
                st.write(f"Genres: {movie['genres']}")
                st.write(f"Release Year: {movie['release_year']}")
                if st.button(f"Rate {movie['title']}", key=f"rate_{movie_id}"):
                    rating = st.number_input('Your rating', min_value=0.5, max_value=5.0, value=3.0, step=0.5, key=f"slider_{movie_id}")
                    st.write(f"You rated {movie['title']} {rating} stars!")
            st.write('---')

    elif page == 'Get Recommendations':
        st.header('🎯 Get Personalized Recommendations')
        num_ratings = 6
        num_recommendations = 5

        # Initialize session state for user ratings and sampled movies
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = [3.0] * num_ratings  # Use float here

        if 'sampled_movies' not in st.session_state:
            st.session_state.sampled_movies = merged_df.sample(num_ratings).reset_index(drop=True)

        # Show all six movies for rating
        for i in range(num_ratings):
            movie = st.session_state.sampled_movies.iloc[i]
            st.write(f"\nMovie: {movie['title']} ({movie['release_year']})")
            st.write(f"Genre: {movie['genres']}")
            st.session_state.user_ratings[i] = st.number_input(
                f"Rate {movie['title']}",
                min_value=0.5,  # Float
                max_value=5.0,  # Float
                value=float(st.session_state.user_ratings[i]),  # Convert to float
                step=0.5,  # Float
                key=f"rating_{movie['movieId']}"
            )

        if st.button('Get Recommendations'):
            model = CollabBasedModel(merged_df, collab_model)
            user_ratings = [(st.session_state.sampled_movies.iloc[i]['movieId'], st.session_state.user_ratings[i]) for i in range(num_ratings)]
            recommendations = model.get_recommendations(user_ratings, n=num_recommendations)

            st.subheader('Your Recommended Movies:')
            for movie_id, score in recommendations:
                movie = merged_df[merged_df['movieId'] == movie_id].iloc[0]
                col1, col2 = st.columns([1, 3])
                with col1:
                    poster_url = fetch_poster(movie['title'])
                    st.image(poster_url, width=150)
                with col2:
                    st.write(f"**{movie['title']}**")
                    st.write(f"Genres: {movie['genres']}")
                    st.write(f"Predicted Rating: {score:.2f}")
                    trailer_url = get_trailer_url(movie['title'])
                    if trailer_url:
                        st.write(f"[Watch Trailer]({trailer_url})")
                    else:
                        st.write("Sorry, couldn't find a trailer for this movie.")
                st.write('---')

    elif page == 'Search Movies':
        st.header('🔍 Search Movies')
        search_term = st.text_input('Enter a movie title')
        if search_term:
            results = merged_df[merged_df['title'].str.contains(search_term, case=False)]
            if results.empty:
                st.write("No movies found matching your search term.")
            else:
                for _, movie in results.iterrows():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        poster_url = fetch_poster(movie['title'])
                        st.image(poster_url, width=150)
                    with col2:
                        st.subheader(movie['title'])
                        st.write(f"Genres: {movie['genres']}")
                        st.write(f"Release Year: {movie['release_year']}")
                        trailer_url = get_trailer_url(movie['title'])
                        if trailer_url:
                            st.write(f"[Watch Trailer]({trailer_url})")
                        else:
                            st.write("Sorry, couldn't find a trailer for this movie.")
                    st.write('---')

        st.subheader('Browse by Genre')
        genres = merged_df['genres'].str.get_dummies(sep=',').columns.tolist()
        genres.insert(0, 'All')  # Add 'All' option to the list
        selected_genre = st.selectbox('Select a Genre', genres)

        if selected_genre:
            st.subheader(f'Top 10 {selected_genre.capitalize()} Movies')
            if selected_genre == 'All':
                genre_results = merged_df.nlargest(10, 'rating')
            else:
                genre_results = merged_df[merged_df['genres'].str.contains(selected_genre)].nlargest(10, 'rating')
            for _, movie in genre_results.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    poster_url = fetch_poster(movie['title'])
                    st.image(poster_url, width=150)
                with col2:
                    st.subheader(movie['title'])
                    st.write(f"Genres: {movie['genres']}")
                    st.write(f"Release Year: {movie['release_year']}")
                    trailer_url = get_trailer_url(movie['title'])
                    if trailer_url:
                        st.write(f"[Watch Trailer]({trailer_url})")
                    else:
                        st.write("Sorry, couldn't find a trailer for this movie.")
                st.write('---')

    elif page == 'About':
        st.header('📚 About Streamflix')
        st.write(
            "Streamflix is a movie recommendation system that combines collaborative filtering and content-based methods "
            "to provide personalized movie suggestions. Our system uses your ratings and movie genres to recommend "
            "movies you might enjoy. Explore top trending movies, get personalized recommendations, and search for "
            "your favorite films all in one place.")

        st.subheader('Developers')
        st.write(
            "- **Evaclaire Wamitu**\n"
            "  - [GitHub](https://github.com/Eva-Claire)\n"
            "  - Email: [evamunyika@gmail.com](mailto:evamunyika@gmail.com)\n\n"
            "- **Simon Makumi**\n"
            "  - [GitHub](https://github.com/simonMakumi)\n"
            "  - Email: [simonmakumi5@gmail.com](mailto:simonmakumi5@gmail.com)"
        )

if __name__ == '__main__':
    main()