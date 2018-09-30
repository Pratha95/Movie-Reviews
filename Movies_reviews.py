
# coding: utf-8

# In[8]:


movies_path = 'D:\MS course work\Visualization engg\ml-latest-small\movies.csv'
import pandas as pd
from pandas import DataFrame as df
movies_df = pd.read_csv(movies_path)
movies_df.head()


# In[11]:


len(movies_df.index)


# In[12]:


movies_df.shape


# In[18]:


movies_df['new_genre'] = movies_df['genres'].str.split('|')
movies_df.head()


# In[39]:


best_lambda = lambda x: set(['Thriller','Romance','Action','Comedy']).issubset(x)
best_movies = movies_df[movies_df.new_genre.map(best_lambda)]
len(best_movies.index)
best_movies


# In[41]:


count_lambda = lambda x: len(x)
movies_df['genre_count'] = movies_df.new_genre.apply(count_lambda)
movies_df.head()


# In[75]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.hist(movies_df.genre_count)
plt.title("genres histogram")
plt.xlabel("# of genres")
plt.ylabel("# of movies")
plt.axis([0,9,0,5000])
plt.grid()
plt.show()


# In[64]:


# to see how many movies per genre
from collections import Counter
flat_genre = [item for sublist in movies_df.new_genre for item in sublist]
genre_dict = dict(Counter(flat_genre))
print (genre_dict)


# In[71]:


# plotting movies per genre using pie chart
plt.pie(genre_dict.values(), labels = genre_dict.keys())
plt.title("Movies per genre distribution")
plt.show()


# In[84]:


ratings_path = 'D:/MS course work/Visualization engg/ml-latest-small/ratings.csv'
ratings_df = pd.read_csv(ratings_path)
ratings_df.head()


# In[111]:


print, len(ratings_df.index)
print , len(ratings_df.userId.unique())
#
print, len(ratings_df.movieId.unique())


# In[104]:


print, len(ratings_df.userId.unique())
print, len(ratings_df.movieId.unique())
ratings_df.rating.unique()


# In[112]:


import numpy as np


# In[114]:


#mean rating
np.mean(ratings_df.rating)


# In[116]:


#to find the most common rating given by the users
from scipy import stats
stats.mode(ratings_df.rating)


# In[120]:


plt.hist(ratings_df.rating)
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
plt.xlabel("rating")
plt.ylabel("no. of movies")
plt.grid()
plt.show()


# In[123]:


#to find the highest rated movies
ratings_view = ratings_df[['movieId', 'rating']]
ratings_view.groupby(['movieId'], as_index=False).mean().sort_values(by='rating', ascending=False).head(10)


# In[126]:


#merging the ratings and movies dataset
merged_df = pd.merge(ratings_df,movies_df, on='movieId')
merged_df.head(5)


# In[129]:


#to get the top listed movies, we have get the titles
title_df = merged_df[['movieId', 'title',  'rating']]
title_df.groupby(['movieId', 'title'], as_index=False).mean().sort_values(by='rating',ascending=False).head(10)


# In[131]:


len(merged_df[merged_df['movieId']==88448].index)


# In[136]:


#let us consider movies with 100+ ratings only
temp_df = title_df.groupby(['movieId','title'],as_index=False).count()
well_rated_df = temp_df[temp_df['rating'] > 100].sort_values(by= 'rating', ascending=False)
well_rated_df.head()


# In[137]:


# now lets created a filtered df from merged_df which only has these movies and then find top 20 movies
filtered_df = merged_df[merged_df['movieId'].apply(lambda x: x in list(well_rated_df['movieId']))]
title_df = filtered_df[['title', 'rating', 'movieId']]
title_df.groupby(['movieId', 'title'], as_index=False).mean().sort_values(by='rating', ascending=False).head(20)


# In[138]:


# now lets add a column called rating_year which depicts the year when the rating was given
import datetime
year_lambda = lambda x: int(datetime.datetime.fromtimestamp(x).strftime('%Y'))
merged_df['rating_year'] = merged_df['timestamp'].apply(year_lambda)
merged_df.head()


# In[139]:


# now lets add a column called rating_year which depicts the year when the rating was given
import datetime
year_lambda = lambda x: int(datetime.datetime.fromtimestamp(x).strftime('%Y'))
merged_df['rating_year'] = merged_df['timestamp'].apply(year_lambda)
merged_df.head()


# In[141]:


# now lets create a new data frame which contains number of ratings given on each year
ratings_per_year = merged_df.groupby(['rating_year'])['rating_year'].count()
ratings_per_year.head(10)


# In[143]:


# now lets get some stats on number of ratings per year

years = ratings_per_year.keys()
num_ratings = ratings_per_year.get_values()
print , np.mean(num_ratings)


# In[144]:


# now lets scatter plot this data to visualize how ratings are spead across years
plt.scatter(years, num_ratings)
plt.title('# of rating across years')
plt.xlabel('Year')
plt.ylabel('# of ratings')
plt.show()


# In[155]:


# now lets try to build a linear regression model using which we will predict how many ratings we get each year
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(years, num_ratings)


# In[157]:


# now lets use the slope and intercept to create a predict function which will predict num_ratings given a year
def predict_num_ratings(year):
    return slope * year + intercept

predicted_ratings = predict_num_ratings(years)


# In[158]:


# now lets plot our predicted values along side the actual data to see how well we did
plt.scatter(years, num_ratings)
plt.plot(years, predicted_ratings, c='r')
plt.show()


# In[160]:


# now lets see how good our prediction is by calculating the r-squared value
r_square = r_value ** 2
print (r_square)


# In[161]:


# now lets try a polynomial function instead of a linear function and see if that fits better
polynomial = np.poly1d(np.polyfit(years, num_ratings, 3))
plt.scatter(years, num_ratings)
plt.plot(years, polynomial(years), c='r')
plt.show()


# In[163]:


# now lets calculate the r-square for this polynomial regression

from sklearn.metrics import r2_score
r2 = r2_score(num_ratings, polynomial(years))
print (r2)


# In[165]:


# now we can predict how many ratings we expect in any year using our polynomial function
print (polynomial(2017))          
print (polynomial(2018))

