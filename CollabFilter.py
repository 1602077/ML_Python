""" Collaborative Filtering - Movies Dataset """
import pandas as pd
import numpy as np
from IPython.display import display
from math import sqrt
import matplotlib.pyplot as plt

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

#Each movie has ID, title & yr, genres in same field - remove year from title column and seperate
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
#Splits into two including (****) now remove ()
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
#remove years from title colum
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#strip function to remove any whitespace
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
#drop genres column as not neccesary for this recommender system
movies_df = movies_df.drop('genres', 1)
#Each row has userID, movieId, rating, timestamp
ratings_df = ratings_df.drop('timestamp', 1)
#display(ratings_df.head())

##############################################################################################
# USER-USER FILTERING - PEARSON CORRELATION FUNCTION
##############################################################################################
"""
Process of creating a user based recommendation system:
1. select user with movies user has watched
2. based on their ratings of movies find top X neighbours
3. get watched movie record of user for each neighbour
4. calculate similarity score using a given formula
5. recommend items with the highestscore
"""
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title': 'Jumanji', 'rating':2},
            {'title':'Pulp Fiction', 'rating':5},
            {'title':'Akira', 'rating':4.5}
            ]
inputMovies = pd.DataFrame(userInput)
#display(inputMovies)
#Filtering movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#merge input id and input movies
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('year', 1)
#Filtering out user that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
#groupby creates sub dfs all w/ same value in col for given parameter
userSubsetGroup = userSubset.groupby(['userId'])
#display(userSubsetGroup.get_group(1130)) #i.e. UserId=1130
#sort so users w/ movie most in common as input have priority
userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

#create subset to iterate over to save time
userSubsetGroup = userSubsetGroup[0:100]
#Store pearsn correleation in a dictionary, where key is the user Id and value is the coeffo
pearsonCorrelationDict = {}
for name, group in userSubsetGroup:
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    nRatings = len(group)
    #Review scores for movies they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #Store them in a temp buffer variable in list format to faciliate further calcs
    tempRatingList = temp_df['rating'].tolist()
    tempGroupList = group['rating'].tolist()
    #Find Pearson correleation for two users, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

#Get top 50 users that are most similar to the input, now start recommending movies to input user
topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
#display(topUsers.head())
topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
#display(topUsersRating.head())
#Mutlipy movie rating by its weight (similarity index), sum up new ratings, and divice by sum of weights

topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
#display(tempTopUsersRating.head())
recommendation_df = pd.DataFrame() #empty df
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index

#Sort to see the top 20 recommended movies
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
display(recommendation_df.head(20))
print( "Recommended Movies: \n", movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(20)['movieId'].tolist())] )
