#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (TMDb movies dataset )
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > **Overview**: For this project I am using TMDb movies dataset.
# This data set contains information about more than 10 thousand movies collected from The Movie Database (TMDb), including user ratings and revenue. It consist of 21 columns such as imdb_id, revenue, budget, vote_count etc.
# 
# >**Question that can analyised from this data set.**
# >1. Movies which had most and least profit
# 2. Movies with largest and lowest budgets
# 3. Movies with most and least earned revenu.
# 4. Movies with longest and shortest runtime values
# 5. Average runtime of all the movies.
# 6. How did the amount of produced films changed over time?
# 7. In which year we had most no.of profitable movies?.
# >-----------------
# >**Question respest to the profitable and ranked movies.**
# >1. Show the successful genres (with respest to the profitable movies).
# 2. The No.of movies for each category (most ranked and earned).
# 3. Most frequent cast (with respest to the profitable movies).
# 4. Most Frequent Cast( most profit and ranked).
# 5. Average budget (with respest to the profitable movies)
# 6. Average revenue (with respest to the profitable movies)
# 7. Average duration of the movie 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # <a id='wrangling'></a>
# ## Data Wrangling
# 
# >After observing the dataset and proposed questions for the analysis we will be keeping only relevent data deleting the unsued data so that we can make our calculation easy and understandable. 
# 

# ### Load and explore the data

# In[2]:


# Load the data
df = pd.read_csv('tmdb-movies.csv')


# In[3]:


df.head(2)


# 
# 
# ###    Obsevations from the data set
# 
# > No unit of currency is mentioned in the dataset. So for my analysis I will take it as dollar as it is the most used international currency.

# In[4]:


df.shape
print('Dataframe contains {} rows and {} columns'.format(df.shape[0],df.shape[1]))


# the dataframe contains 10866 rows and 21 columns

# ### Get more information about the data
# >the dataframe contains integer, float and string values.
# And there are some columns that contain Null values

# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


print(list(df.columns))


# 
# ### Data Cleaning (Removing the unused information from the dataset)
# > 1. We need to remove unused column such as id, imdb_id, vote_count, production_company, keywords, homepage etc.
# > 2. Removing the duplicacy in the rows(if any).
# > 3. Some movies in the database have zero budget or zero revenue, that is there value has not been recorded so we will be discarding such entries
# > 4. Changing release date column into date format.
# > 5. Check runtime column and replacing zero with NAN if exist.
# > 6. Changing format of budget and revenue column.

# #### 1-Removing Unused columns
# >Columns that we need to delete are - id, imdb_id, popularity, budget_adj, revenue_adj, homepage, keywords, overview, production_companies,and vote_count

# In[8]:


#creating a list of columns to be deleted
del_col=[ 'id', 'imdb_id', 'popularity', 'budget_adj', 'revenue_adj', 'homepage', 'keywords', 'overview', 'production_companies', 'vote_count']

#deleting the columns
df = df.drop(del_col,1)

#show the new dataset
df.head()


# #### 2-Removing the duplicacy in the rows(if any).

# In[9]:


rows, col = df.shape
#We need to reduce the count of row by one as it contain header row also.
print('There are {} total entries of movies and {} no.of columns in it.'.format(rows-1, col))


# > Now remove the duplicated

# In[10]:


#get the number of duplicated rows 

df.drop_duplicates(keep ='first', inplace=True)
no_row = df.shape[0]
print("duplicated rows which deleted = {}".format(rows-no_row))


# #### 3- Removing 0's from budget and the revenue columns
# >first replace 0 value with NaN, then remove the rows which contain NaN for budget and the revenue columns.

# In[11]:


# creating a seperate list of revenue and budget column
temp_list=['budget', 'revenue']

#this will replace all the value from '0' to NAN in the list
df[temp_list] = df[temp_list].replace(0, np.NAN)

#Removing all the row which has NaN value in temp_list 
df.dropna(subset = temp_list, inplace = True)

rows, col = df.shape
print('After removing, we now have only {} no.of movies.'.format(rows-1))


# >now the dataset contain 3853 rows and 11 columns

# #### 4-Changing the release date column into standard date format

# In[12]:


df.release_date = pd.to_datetime(df['release_date'])


# In[13]:


#check the format of release_date
df.head(2)


# #### 5-check the values of runtime column 

# In[14]:


#check the validity of runime column values
df.query('runtime <= 0').count()['runtime']


# >All the values of runtime column are valid. There is no zeros to be replaced 

# #### 6-Changing format of budget and revenue column.

# >check the current types of columns

# In[15]:


df.dtypes


# >change the data type then check the types

# In[16]:


change_type=['budget', 'revenue']
#changing data type
df[change_type]=df[change_type].applymap(np.int64)
#printing the changed information
df.dtypes


# In[17]:


df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# **Calculating the profit of the each movie**

# In[18]:


#insert new column profit= revenue-budget
df.insert(2,'profit_earned',df['revenue']-df['budget'])
df.head(2)


# 
# **create function to get the most and least value in every column**

# In[19]:


#method to calculate the most and least value
def most_least(column):
    most_value = df[column].max()
    least_value = df[column].min()
    most = df.loc[df[column]== most_value]
    least = df.loc[df[column]== least_value]
    return pd.concat([most, least])


# ### Question 1 : Movies which had most and least profit

# In[20]:


#calculate most and least profit_earned
most_least('profit_earned')


# >Column with id 1386 shows the highest earned profit i.e 2544505847 .
# >Whereas the column with id 2244 shows the lowest earned profit i.e -413912431.
# 

# ### Question 2 : Movies with largest and lowest budgets

# In[21]:


most_least('budget')


# >Column with id 2244 shows the highest budget i.e 425000000  .Whereas the columns with id 2618 and 3581  shows the lowest budget i.e 1.

# ### Question 3: Movies with most and least earned revenu.

# In[22]:


most_least('revenue')


# >Column with id 1386 shows the highest revenue i.e 2781505847  .Whereas the columns with id 5067 and 8142  shows the lowest revenue i.e 2.

# ### Question 4: Movies with longest and shortest runtime values

# In[23]:


most_least('runtime')


# >Column with id 2107 shows the longest runtime i.e 338 minutes. Whereas the columns with id 5162  shows the shortest runtime i.e 15 minutes.

# ### Question 5: Average runtime of all the movies.

# In[24]:


# find the mean of runtime column
df['runtime'].mean()


# >the average runtime is 109 minutes

# #### visualize the runtime values

# In[25]:


#plotting a histogram of runtime of movies

#giving the figure size(width, height)
plt.figure(figsize=(10,5), dpi = 100)
 
plt.xlabel('Runtime of the Movies', fontsize = 10) 
plt.ylabel('No.of Movies in the Dataset', fontsize=10)
plt.title('Runtime of all the movies', fontsize=25, color='red')

plt.hist(df['runtime'], rwidth = 0.9, bins = 30)
plt.grid(axis='y', alpha=0.75)
plt.show()


# > - The distribution of the above formed graph is positively skewed or right skewed.
# > - Most of the movies are timed between 80 to 115 minutes. Almost 1000 and more no.of movies fall in this criteria.

# In[26]:


#show the box plot for runtime column
plt.figure(figsize=(10,5))
plt.boxplot(df['runtime'], sym='rs', vert=False, widths=[0.75])
plt.show()


# In[27]:


#runtime column statistics 
df['runtime'].describe()


# >**By looking at both the plot and calculations, we can conclude that:**
# >  - 25% of movies have a runtime of less than 95 minutes
# >  - 50% of movies have a runtime of less than 109 minutes. (median)
# >  - 75% of movies have a runtime of less than 119 minutes
# 

# ### Question 6: How did the amount of produced films changed over time?

# In[28]:


#create a list of releas years without duplicates
years = list(df['release_year'].unique())
#sort the years in ascending order
years.sort()
#dict to store the years and no.of movies
movies={}


# >**for loop** to count no.of movies for every year and store them in a dictionary

# In[29]:


#count no.of movies per year
for movie in years:
    movies[movie]= df['release_year'][df['release_year']== movie].count()


# In[30]:


#create dataframe contains no.of movies per year
movies_year = pd.DataFrame(list(movies.items()), columns=['year','movies'])


# In[31]:


#rotate the data frame to save space
movies_year.T


# In[32]:


#plot the no.of movies per year
#plt.figure(figsize=(10,5))
movies_year.plot(x='year', y='movies', kind='line', figsize=(15,5), label="Amount of movies")
plt.title("Amount of movies over years",fontsize=25, color='red')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Amount', fontsize=15)
plt.legend()
plt.show()


# >The figure shows that the number of movies is increasing over the years but decreased after 2011

# ### Question 7: In which year we had most no.of profitable movies?.

# In[33]:


#Since we want to know the profits of movies for every year  
#therefore we have to sum up all the movies of a particular year

profits_year = df.groupby('release_year')['profit_earned'].sum()

#plotting the graph
plt.figure(figsize=(15,5))
plt.xlabel('Release Year of Movies in the data set', fontsize = 16)
plt.ylabel('Profits earned by Movies', fontsize = 16)
plt.title('Representing Total Profits earned by all movies Vs Year of their release.', fontsize = 20, color='red')
plt.plot(profits_year)
plt.show()


# In[34]:


#To find that which year made the highest profit?
profits_year.idxmax()


# >So  **2015** was the year where movies made the highest profit.

# ## Questions with respect to the profitable and ranked
# 
# > - Before moving further we need to clean our data again. We will be considering only those movies who have earned a significant amount of profit.So lets fix this amount to 60 million dollar and store them in a dataframe
# > - get movies which have earned a significant amount of profit greater than or equal 60 million and vote average 7 or greater and store them in another dataframe

# In[35]:


#selecting the movies having profit $60M or more
profit_data = df[df['profit_earned'] >= 60000000]
#selecting the movies having profit $60M or more and ranking 7
rank_data = profit_data[profit_data['vote_average'] >= 7.0]

#reindexing new data
profit_data.index = range(len(profit_data))

#we will start from 1 instead of 0
profit_data.index = profit_data.index + 1

#printing the changed dataset
profit_data.head(3)


# In[36]:


#counting the no.of rows in the new data frame
len(profit_data)


# In[37]:


len(rank_data)


# >separate the column that have multiple values as a string (such as cast and genres) into several columns to applay the calculation

# In[38]:


#function which will take a column and dataframe as arguments and keep its track 
def data(column, dataframe):
    #will take a column, and separate the string by '|'
    data = dataframe[column].str.cat(sep = '|')
    
    #giving pandas series and storing the values separately
    data = pd.Series(data.split('|'))
    
    #arranging in descending order
    count = data.value_counts(ascending = False)
    
    return count


# In[39]:


#variable to store the retured value
count = data('genres', profit_data)
#printing top 5 values
count.head()


# ### Q1- Show the successful genres

# In[40]:


#create a dict that contains categories as keys and no.of movies as values
category_count={}
for cat in count.index:
    category_count[cat] = count[cat]


# In[41]:


#get the keys for x-axis
x = list(category_count.keys())
#get the values for y-axis
y = list(category_count.values())


# In[42]:



plt.figure(figsize=(15,5))
plt.xlabel('Categories', fontsize = 20) 
plt.ylabel('No.of Movies', fontsize=20)
plt.title('No.of Movies for each category', fontsize=25, color='red')

plt.bar(x,y)
plt.xticks(rotation= 45 ,color='red')
plt.show()


# ### Q2- The No.of movies for each category (most ranked and earned)
# >this for movies ranked 7 or more and amount of profit 60 Million or more

# In[44]:


data('genres', rank_data)


# ### Q3- Most Frequent Cast( most profit)

# In[45]:


#variable to store the retured value
count = data('cast', profit_data)
#printing top 5 values
count.head()


# ### Q4- Most Frequent Cast( most ranked)

# In[46]:


count = data('cast', rank_data)
#printing top 5 values
count.head()


#  ### Q5- Average Budget of the movies

# >Create function to find average

# In[47]:


#New function to find average 
def profit_avg(column):
    return profit_data[column].mean()


# In[48]:


# calling the above function for budget
profit_avg('budget')


# >The movies having profit of 60 million dollar and more have an average budget of 63 million dollar.

# ### Q6- Average Revenue earned by the movies

# In[49]:


#calling the above function for revenue
profit_avg('revenue')


# >The movies having profit of 60 million dollar and more have an average revenue of 274 million dollar.

# ### Q7- Average duration of the movies

# In[50]:


# calling the above function for 
profit_avg('runtime')


# >So the movies having profit of 60 million dollar and more have an average duration of 114 minutes.

# <a id='conclusions'></a>
# ## Conclusions:
# 
# > - In order to ensure the highest rank :
#      - Any one of these should be in the cast : Tom Hanks,Brad Pitt,Leonardo DiCaprio,Emma Watson,Matt Damon.
#      - Genre must be : Drama,Action,Thriller,Adventure,Comedy.
# 
# > - For a Movie to be in successful criteria :
#     - Average Budget must be around 63.7 millon dollar
#     - Average duration of the movie must be 114 minutes
#     - Any one of these should be in the cast : Tom Cruise, Brad Pitt,Tom Hanks,Sylvester Stallone,Cameron Diaz  
#     - Genre must be : Comedy, Action,Drama, Thriller,  Adventure.
# > - <font color= blue>**By doing all this the movie might be one of the hits and hence can earn an average revenue of around 274.7 million dollar.**</font>
# ***
# ### Limitations:
#    - This analysis was done considering the movies which had a significant amount of profit of around 60 million dollar.
#    - This might not be completely error free but by following these suggestion one can increase the probability of a movie to become a hit.
#    - Moreover we are not sure if the data provided to us is completel corect and up-to-date. 
#    - As mentioned before the budget and revenue column do not have currency unit, it might be possible different movies have budget in different currency according to the country they are produce in. 
#    - So a disparity arises here which can state the complete analysis wrong. Dropping the rows with missing values also affected the overall analysis.
