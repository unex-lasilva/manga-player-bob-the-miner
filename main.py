import pandas as pd
import ast

ratings = pd.read_csv('data_set/ratings_small.csv')
metadata = pd.read_csv('data_set/movies_metadata.csv', low_memory=False)

ratings = ratings[['userId', 'movieId', 'rating']]
metadata = metadata[['id', 'title', 'genres']]

metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
metadata = metadata.rename(columns={'id': 'movieId'})
metadata = metadata.dropna(subset=['movieId'])
metadata['movieId'] = metadata['movieId'].astype(int)

data_frame = pd.merge(ratings, metadata, on='movieId')

data_frame['genres'] = data_frame['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
data_frame['genres'] = data_frame['genres'].apply(lambda genre_list: [genre['name'] for genre in genre_list])

data_frame['liked'] = data_frame['rating'] >= 4.0

all_genres = set(g for lista in data_frame['genres'] for g in lista)
for genre in all_genres:
    data_frame[genre] = data_frame['genres'].apply(lambda genres: 1 if genre in genres else 0)

print(data_frame.head())
