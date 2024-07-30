import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
import requests
import pickle

# Set page config
st.set_page_config(page_title='STREAMFLIX', page_icon="🎬", layout='wide')

# Load your data
@st.cache_data
def load_data():
    df = pd.read_csv('movies_data/movies.csv')
    ratings = pd.read_csv('movies_data/ratings.csv')
    return df, ratings

# Train your model
@st.cache_resource
def train_model(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId_x', 'movieId', 'rating']], reader)
    model = SVD()
    model.fit(data.build_full_trainset())
    return model

# Get recommendations
def get_recommendations(model, df, user_ratings, n=5, genre=None):
    new_user_id = df['userId_x'].max() + 1
    movies_to_predict = df[~df['movieId'].isin([x[0] for x in user_ratings])]['movieId'].unique()
    
    predictions = []
    for movie_id in movies_to_predict:
        predicted_rating = model.predict(new_user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))
    
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    if genre:
        genre_recommendations = [
            (movie_id, rating) for movie_id, rating in recommendations
            if genre.lower() in df[df['movieId'] == movie_id]['genres'].iloc[0].lower()
        ]
        return genre_recommendations[:n]
    else:
        return recommendations[:n]

# Fetch movie poster
@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=your_api_key"
    response = requests.get(url)
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', '')

# Main app
def main():
    st.title("🎬 Streamflix: Hybrid Movie Recommendation System")

    # Load data
    df, ratings = load_data()
    model = train_model(ratings)

    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home', 'Get Recommendations', 'Search Movie'])

    if page == 'Home':
        st.header('🔥 Top Trending Movies')
        top_movies = df.sort_values('popularity', ascending=False).head(10)
        
        for _, movie in top_movies.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                poster_url = fetch_poster(movie['id'])
                st.image(poster_url, width=150)
            with col2:
                st.subheader(movie['title'])
                st.write(f"Genres: {movie['genres']}")
                st.write(f"Average Rating: {movie['vote_average']:.1f}/10")
                if st.button(f"Rate {movie['title']}", key=f"rate_{movie['id']}"):
                    rating = st.slider('Your rating', 0.5, 5.0, 3.0, 0.5, key=f"slider_{movie['id']}")
                    st.write(f"You rated {movie['title']} {rating} stars!")
            st.write(''---'')

    elif page == 'Get Recommendations':
        st.header('🎯 Get Personalized Recommendations')
        user_id = st.number_input('Please enter your user ID', min_value=1, step=1)
        genres = st.multiselect('Select genres', df['genres'].explode().unique())
        
        if st.button('Get Recommendations'):
            recommendations = get_recommendations(user_id, model, df, ratings)
            if genres:
                recommendations = recommendations[recommendations['genres'].apply(lambda x: any(genre in x for genre in genres))]
            
            st.subheader('Your Recommended Movies:')
            for _, movie in recommendations.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    poster_url = fetch_poster(movie['id'])
                    st.image(poster_url, width=150)
                with col2:
                    st.write(f'**{movie['title']}**')
                    st.write(f'Genres: {movie['genres']}')
                    st.write(f'Average Rating: {movie['vote_average']:.1f}/10')
                    if st.button(f'Watch Trailer for {movie['title']}', key=f'trailer_{movie['id']}'):
                        # You would need to implement a function to fetch and display the trailer
                        st.video('https://www.youtube.com/watch?v=dQw4w9WgXcQ')  # Placeholder
                st.write('---')

    elif page == 'Search Movies':
        st.header('🔍 Search Movies')
        search_term = st.text_input('Enter a movie title')
        if search_term:
            results = df[df['title'].str.contains(search_term, case=False)]
            for _, movie in results.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    poster_url = fetch_poster(movie['id'])
                    st.image(poster_url, width=150)
                with col2:
                    st.subheader(movie['title'])
                    st.write(f'Genres: {movie['genres']}')
                    st.write(f'Average Rating: {movie['vote_average']:.1f}/10')
                    st.write(f'Overview: {movie['overview'][:200]}...')
                st.write('---')

if __name__ == '__main__':
    main()