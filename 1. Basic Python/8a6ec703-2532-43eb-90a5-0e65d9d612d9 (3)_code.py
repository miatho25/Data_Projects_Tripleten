# In[4]
# importing pandas
import pandas as pd

# In[6]
# reading the files and storing them to df
mv_and_shows = '/datasets/movies_and_shows.csv'
df = pd.read_csv('/datasets/movies_and_shows.csv')

# In[8]
# obtaining the first 10 rows from the df table
# hint: you can use head() and tail() in Jupyter Notebook without wrapping them into print()
df.head(10)

# In[10]
# obtaining general information about the data in df
df.info()

# In[13]
# the list of column names in the df table
column_names = df.columns
column_names

# In[15]
# renaming columns
df = df.rename(columns={'   name': 'name', 'Character': 'character', 'r0le': 'role', 'TITLE': 'title', '  Type': 'type', 'release Year': 'release_year', 'genres': 'genres', 'imdb sc0re': 'imdb_score', 'imdb v0tes': 'imdb_votes'})

# In[17]
# checking result: the list of column names
df.columns

# In[19]
# calculating missing values
df.isna().sum()

# In[21]
# dropping rows where columns with scores, and votes have missing values
df = df.dropna()
df

# In[23]
# counting missing values
df.isna().sum()

# In[25]
# counting duplicate rows
duplicates = df.duplicated().sum()
duplicates

# In[27]
# removing duplicate rows
df = df.drop_duplicates()

# In[29]
# checking for duplicates
df

# In[32]
# viewing unique type names
df['type'].unique()
unique_types = sorted(df['type'].unique())
unique_types

# In[34]
# function for replacing implicit duplicates
def replace_wrong_show(wrong_shows_list, correct_show):
    df['type'] = df['type'].replace(wrong_shows_list, correct_show)
wrong_shows = ['shows', 'SHOW', 'tv show', 'tv shows', 'tv series', 'tv']

# In[36]
# removing implicit duplicates
replace_wrong_show(wrong_shows, 'show')

# In[38]
# viewing unique genre names
unique_types = sorted(df['type'].unique())
unique_types

# In[43]
# using conditional indexing modify df so it has only titles released after 1999 (with 1999 included)
# give the slice of dataframe new name
data_filtered = df[df['release_year'] >= 1999]
data_filtered

# In[44]
# repeat conditional indexing so df has only shows (movies are removed as result)
df_filtered = df[df['type'] == 'show']
df_filtered

# In[46]
# rounding column with scores
df_shows_1999_plus = df.loc[(df['release_year'] >= 1999) & (df['type'] == 'show')].copy()
df_shows_1999_plus.loc[:, 'score_bucket'] = df_shows_1999_plus['imdb_score'].round(0).astype(int)
#checking the outcome with tail()
df_shows_1999_plus[['imdb_score', 'score_bucket']].tail()

# In[48]
# Use groupby() for scores and count all unique values in each group, print the result
score_counts = df_shows_1999_plus.groupby('score_bucket')['imdb_votes'].count()
score_counts

# In[50]
df_filtered = df_shows_1999_plus[~df_shows_1999_plus['score_bucket'].isin([2, 3, 10])]
print(sorted(df_filtered['score_bucket'].unique()))

# In[52]
# filter dataframe using two conditions (scores to be in the range 4-9)
df_filtered = df_shows_1999_plus[(df_shows_1999_plus['score_bucket'] >= 4) & (df_shows_1999_plus['score_bucket'] <= 9)]
# group scores and corresponding average number of votes, reset index and print the result
average_votes = df_filtered.groupby('score_bucket')['imdb_votes'].mean().reset_index()
average_votes

# In[54]
# round column with averages
average_votes['imdb_votes'] = average_votes['imdb_votes'].round(0).astype(int)
# rename columns
average_votes.rename(columns={'score_bucket': 'IMDB Score', 'imdb_votes': 'Average Votes'}, inplace=True)
average_votes_sorted = average_votes.sort_values(by='IMDB Score', ascending=False)
# print dataframe in descending order
average_votes_sorted

