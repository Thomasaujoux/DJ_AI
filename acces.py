############# Informations ###############
# Ce fichier a été fait à partir du projet : https://github.com/makispl/Spotify-Data-Analysis/blob/master/Spotify_Data_Manipulation_Python.ipynb
# Regarder ce tuto pour mieux comprendre : https://stmorse.github.io/journal/spotify-api.html
# Ce tuto à compléter en fonction de ce que l'on veut faire : https://spotipy.readthedocs.io/en/2.22.1/

# Ce fichier est pour lire des informations sur Spotify




######### Import the libraries #########
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials



######### Declare the credentials #########
# My ID's
cid = '92278466ab914b5a8cd997ef7ad26bf0'
secret = '42a4144330f2469e98bb6e50bb276cb5'
redirect_uri='http://localhost:7777/callback'
username = '31xl6hkcysxjzwhu37gl7prssx7i'



######### Authorization flow #########
# A changer en fonction de ce que l'on veut, pour ce fichier on veut lire des informations
scope = 'user-top-read'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)





######### Spotipy manipulations #########

def fetch_playlists(sp, username):
    """
    Returns the user's playlists.
    """
        
    id = []
    name = []
    num_tracks = []
    
    # Make the API request
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        id.append(playlist['id'])
        name.append(playlist['name'])
        num_tracks.append(playlist['tracks']['total'])

    # Create the final df   
    df_playlists = pd.DataFrame({"id":id, "name": name, "#tracks": num_tracks})
    return df_playlists

# Exemple 
playlists = fetch_playlists(sp,username)
playlists = playlists[:4].copy()
playlists

sp = sp
username_id = '31xl6hkcysxjzwhu37gl7prssx7i'
fetch_playlists(sp,username)





def fetch_playlist_tracks(sp, username, playlist_id):
    """
    Returns the tracks for the given playlist.
    """
        
    offset = 0
    tracks = []
    
    # Make the API request
    while True:
        content = sp.user_playlist_tracks(username, playlist_id, fields=None, limit=100, offset=offset, market=None)
        tracks += content['items']
        
        if content['next'] is not None:
            offset += 100
        else:
            break
    
    track_id = []
    track_name = []
    artist_id = []
    artist_info = []
    artist_name = []
    artist_pop = []
    artist_genres = []
    album = []
    track_pop = []
    
    for track in tracks:
        track_id.append(track['track']['id'])
        track_name.append(track['track']['name'])
        artist_id.append(track["track"]["artists"][0]["uri"])
        artist_id2 = track["track"]["artists"][0]["uri"]
        artist_info = sp.artist(artist_id2)
        artist_name.append(track["track"]["artists"][0]["name"])
        artist_pop.append(artist_info["popularity"])
        artist_genres.append(artist_info["genres"])
        album.append(track["track"]["album"]["name"])
        track_pop.append(track["track"]["popularity"])
    
    # Create the final df
    df_playlists_tracks = pd.DataFrame({"track_id":track_id, "track_name": track_name, "artist_id": artist_id, "artist_name": artist_name, "artist_pop": artist_pop, "artist_genres": artist_genres, "album": album, "track_pop": track_pop})
    return df_playlists_tracks

# Exemple
fetch_playlist_tracks(sp, username, '37SqXO5bm81JmGCiuhin0L')




def fetch_audio_features(sp, username, playlist_id):
    """
    Returns the selected audio features of every track, 
    for the given playlist.
    """
    
    # Use the fetch_playlist_tracks function to fetch all of the tracks
    playlist = fetch_playlist_tracks(sp, username, playlist_id)
    index = 0
    audio_features = []
    
    # Make the API request
    while index < playlist.shape[0]:
        audio_features += sp.audio_features(playlist.iloc[index:index + 50, 0])
        index += 50
    
    # Append the audio features in a list
    features_list = []
    for features in audio_features:
        features_list.append([features['acousticness'], features['danceability'], features['duration_ms'],
                              features['energy'], features['instrumentalness'], features['key'], features['liveness'], features['loudness'], features['mode'], features['speechiness'],  features['tempo'],
                              features['time_signature'], features['valence']])
    
    df_audio_features = pd.DataFrame(features_list, columns=['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence'])

    # Set the 'tempo' & 'loudness' in the same range with the rest features
    for feature in df_audio_features.columns:
        if feature == 'tempo' or feature == 'loudness':
            continue
        df_audio_features[feature] = df_audio_features[feature] * 100
    
    # Create the final df, using the 'track_id' as index for future reference
    df_playlist_audio_features = pd.concat([playlist, df_audio_features], axis=1)
    df_playlist_audio_features.set_index('track_id', inplace=True, drop=True)
    
    return df_playlist_audio_features

# Exemple
df_dinner = fetch_audio_features(sp, username, '37SqXO5bm81JmGCiuhin0L')
df_party = fetch_audio_features(sp, username, '2m75Xwwn4YqhwsxHH7Qc9W')
df_lounge = fetch_audio_features(sp, username, '6Jbi3Y7ZNNgSrPaZF4DpUp')
df_pop = fetch_audio_features(sp, username, '3u2nUYNuI08yUg877JE5FI')


df_dinner.head().iloc[:, 1:]



