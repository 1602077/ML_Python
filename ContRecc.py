""" Content-Based Recommender Systems """
import numpy as np
import pandas as pd
from IPython.display import display
from math import sqrt
import matplotlib.pyplot as plt

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
#Removing title and replacing w/ year
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand = False)
#Removing parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand = False)
#Removing years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Appyling strip function to remove white space
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
#Removing seperation  all genres with a '|'
movies_df['genres'] = movies_df.genres.str.split('|')
#display(movies_df.head(4))

#Copying movie df into new one as do not need genres for the first case
moviesWithGenres_df = movies_df.copy()
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
ratings_df = ratings_df.drop('timestamp', 1) #drops timestamp

###############################################################
#CONTENT BASED RECOMMENDATION SYSTEM
###############################################################

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':'Pulp Fiction', 'rating':5},
            {'title':'Akira', 'rating':4.5}
        ]
#Above are the movies that the user likes/has watched.
inputMovies = pd.DataFrame(userInput)
#Extract input movies ID from df and drop unnecessary columns to save memory
#Filter movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist()) ]
inputMovies = pd.merge(inputId, inputMovies)
#Drop unuseful info in the df
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist()) ]
#print("User Movies:\n", userMovies)

#Only need genre table: reset index, drop movield, title, genres and yr colums
userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print("User Genre Table \n", userGenreTable)
