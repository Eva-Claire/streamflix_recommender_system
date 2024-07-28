# STREAMFLIX MOVIE RECOMMENDATION SYSTEM

![attachment:logo.png](logo.png)

### Authors - Group 10
Evaclaire Wamitu - [Email](evamunyika@gmail.com)
Simon Makumi - [Email](simonmakumi5@gmail.com)

## BUSINESS UNDERSTANDING

### Overview
StreamFlix is developing a personalized movie recommendation system to enhance user experience and retention. The system uses collaborative filtering and content based filtering on the MovieLens dataset to provide tailored top 5 movie suggestions for each user. The system will analyze user ratings to generate recommendations using collaborative filtering techniques. New users will be onboarded through various rating collection methods to quickly build their preference profiles. By implementing this system, StreamFlix aims to create a more engaging and personalized viewing experience, ultimately leading to increased user satisfaction and improved business metrics.

### Business Problem
StreamFlix is facing challenges with user retention and engagement. Users are also overwhelmed by the vast library of movies available and often spend a considerable amount of time searching for movies they would enjoy. StreamFlix is, therefore, looking for a way to provide personalized movie recommendations to its users to improve their viewing experience and increase platform usage.

### Objectives

#### Main Objective
To develop and deploy a collaborative filtering-based recommendation system that accurately predicts user preferences and provides relevant movie suggestions.

#### Specific Objectives
1. To build a collaborative filtering model that uses user ratings to generate top 5 movie recommendations.
2. To address the cold start problem using content-based filtering for new users.
3. To evaluate the recommendation system using appropriate metrics like RMSE and MAP.

## DATA UNDERSTANDING
The data utilised in this project is the Movielens dataset from GroupLens Research Lab covering movie ratings from 1902 to 2018. The dataset contains 100836 ratings and 3683 tag applications across 9742 movies with each user rating at least 20 films. While the full dataset contains 1.9 million ratings, we focussed on a subset of about 100,000 for our current model due to time and resource constraints. This sample size balances computational efficiency with statistical relevance for our recommendation engine development. The datasets include `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`. The following features were utilized  in the development of our recommendation system:  `movieId`, `userId_x`, `rating`, `title` and `genres`. 
The datasets were merged on the movieId column resulting in a DataFrame with 285783  rows and 11 columns. The data was sufficient in fulfilling our objectives although additional information such as actors, directors, production studio, runtime and user demographics would have provided more context and insights into user preferences and movie characteristics leading to better recommendations.

## Observations
We decided  to investigate the distribution of ratings and genres and the following

![attachment:logo.png](logo.png)

 