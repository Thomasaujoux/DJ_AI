


############# Informations ###############
# Regarder ce tuto pour mieux comprendre : https://stmorse.github.io/journal/spotify-api.html
# Ce tuto à compléter en fonction de ce que l'on veut faire : https://spotipy.readthedocs.io/en/2.22.1/




############# Les identifiants à changer sur le DashBoard ###############
############# Identifiants limités à 200 requetes ###############
# My ID's
CLIENT_ID = '0dcb28d2945241c481bc7023b0b7d6e1'
CLIENT_SECRET = '4831fa5bf30b4a6a98804bbb90f72274'



############# Librairies bases de données ###############
import pandas as pd
import matplotlib.pyplot as plt
from pandas import json_normalize
import map



############# Librairies et importations pour Spotipy ###############
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)



############# Librairies et importations pour Spotify API ###############
import requests

AUTH_URL = 'https://accounts.spotify.com/api/token'

# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# convert the response to JSON
auth_response_data = auth_response.json()

# save the access token
access_token = auth_response_data['access_token']

headers = {'Authorization': 'Bearer {token}'.format(token=access_token)}

# base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'






############# Avoir les features pour un son avec son id ###############
# Get audio feature information for a single track identified by its unique Spotify ID.
def get_info(track_id):
    # [acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]
    # Description des différentes features : https://developer.spotify.com/documentation/web-api/reference/get-audio-features
    # actual GET request with proper header
    r = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    r = r.json()
    r = json_normalize(r)
    r = r[['id','acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
    return r

track_id = "11dFghVXANMlKmJXsNCbNl"
info = get_info(track_id)






############# Importation de playlists ###############
# A partir de l'ID d'une playlist trouver les sons qui sont dedans et leurs features
def get_playlist_tracks(username,playlist_id):
    # username : client_credentials_manager
    # playlist_id : "https://open.spotify.com/playlist/67P7kU1wEfx1oCQUmsVWQu"
    # playlist_id est données sous le format d'un lien https vers la playlist
    # Les techniques usuelles s'arrêtent à 100 musiques par playlist, cette manipulation permet d'accèder à toutes les musiques.
    # le résultat est un dataframe pour chaque sons de la playlist avec les infos : ["track_uri", "track_name", "artist_uri","artist_info","artist_name","artist_pop","artist_genres","album",'track_pop','acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
    
    # utilisation de spotipy pour importer les informations concernant la playlist
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    
    # Boucle while pour importer toutes les musiques de la plylist sans être limité à 100
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    # Construction du dataframe
    col_names =  ["track_uri", "track_name", "artist_uri","artist_info","artist_name","artist_pop","artist_genres","album",'track_pop','acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
    df  = pd.DataFrame(columns = col_names)
    
    # Boucle pour trouver les features sur l'ensemble des sons de la playlist
    for track in tracks:
        #URI
        track_uri = track["track"]["uri"]
        #Track name
        track_name = track["track"]["name"]
        #Main Artist
        artist_uri = track["track"]["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)
        #Name, popularity, genre
        artist_name = track["track"]["artists"][0]["name"]
        artist_pop = artist_info["popularity"]
        artist_genres = artist_info["genres"]
        #Album
        album = track["track"]["album"]["name"]
        #Popularity of the track
        track_pop = track["track"]["popularity"]
        features = sp.audio_features(track_uri)[0]
        features = json_normalize(features)
        features = features[['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]
        new_row = pd.DataFrame([[track_uri, track_name, artist_uri, artist_info, artist_name, artist_pop, artist_genres, album, track_pop]], columns = ["track_uri", "track_name", "artist_uri","artist_info","artist_name","artist_pop","artist_genres","album",'track_pop'])
        
        df_new_row = pd.concat([new_row, features], axis=1)
        df = pd.concat([df,df_new_row], ignore_index=True)
    df[['duration_ms']] = df[['duration_ms']].apply(pd.to_numeric)
    df[['track_pop']] = df[['track_pop']].apply(pd.to_numeric)
    df[['key']] = df[['key']].apply(pd.to_numeric)
    df[['mode']] = df[['mode']].apply(pd.to_numeric)
        
    return df



def scrap_playlist_tracks(username,playlist_ids):
    L = []
    for i in playlist_ids:
        print(i)
        L.append(get_playlist_tracks(username,i))
    return L
        
username = client_credentials_manager
playlist_ids = ["https://open.spotify.com/playlist/67P7kU1wEfx1oCQUmsVWQu", "https://open.spotify.com/playlist/5xS3Gi0fA3Uo6RScucyct6", "https://open.spotify.com/playlist/13bvOZl3gV7si5jRHpBzeI"]
data = scrap_playlist_tracks(username,playlist_ids)






############# Obtenir le son le plus proche avec les paramètres ###############
# A partir d'une playlist trouver le son le plus adapté
def similar_song():









############# Obtenir la recommendation selon les features ###############
genre_seed = requests.get(BASE_URL + 'recommendations/available-genre-seeds', headers=headers)
genre_seed = genre_seed.json()



artist_uri = '4NHQUGzhtTLFvgF5SZesLK'
seed_genres = 'country'
seed_tracks = "0c6xIDDpzE81m2q797ordA"
r = requests.get(BASE_URL + 'recommendations/', headers=headers, limit = 10, seed_artists = artist_uri, seed_genres = seed_genres, seed_tracks = seed_tracks)

def get_reco(artist_uri,seed_genres, seed_tracks):
    r = requests.get(BASE_URL + 'recommendations/', headers=headers, params={'seed_artists': artist_uri, 'albumseed_genres' : seed_genres, 'seed_tracks': seed_tracks})
    r = r.json()
    track = r
    
    col_names =  ["track_uri", "track_name", "artist_uri","artist_info","artist_name","artist_pop","artist_genres","album",'track_pop','acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
    df  = pd.DataFrame(columns = col_names)
    for i in range(len(r['tracks'])):
    
        track_uri = track["tracks"][i]["uri"]
        #Track name
        track_name = track["tracks"][i]["name"]
        #Main Artist
        artist_uri = track["tracks"][i]["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)
        
        #Name, popularity, genre
        artist_name = track["tracks"][i]["artists"][0]["name"]
        artist_pop = artist_info["popularity"]
        artist_genres = artist_info["genres"]
        
        #Album
        album = track["tracks"][i]["album"]["name"]
        
        #Popularity of the track
        track_pop = track["tracks"][i]["popularity"]
        
        features = sp.audio_features(track_uri)[0]
        acousticness = features["acousticness"]
        danceability = features["danceability"]
        duration_ms = features["duration_ms"]
        energy = features["energy"]
        instrumentalness = features["instrumentalness"]
        key = features["key"]
        liveness = features["liveness"]
        loudness = features["loudness"]
        mode = features["mode"]
        speechiness = features["speechiness"]
        tempo = features["tempo"]
        time_signature = features["time_signature"]
        valence = features["valence"]
        df_new_row =  pd.DataFrame([[track_uri, track_name, artist_uri, artist_info, artist_name, artist_pop, artist_genres, album, track_pop,acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]], columns = col_names)
        df = pd.concat([df,df_new_row], ignore_index=True)
    return df

a = get_reco(artist_uri,seed_genres, seed_tracks)
        
params={'include_groups': 'album', 'limit': 50}








