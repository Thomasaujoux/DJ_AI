

############# Informations ###############
# Regarder ce tuto pour mieux comprendre : https://stmorse.github.io/journal/spotify-api.html
# Ce tuto à compléter en fonction de ce que l'on veut faire : https://spotipy.readthedocs.io/en/2.22.1/




############# Librairies ###############
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials



#Import Library
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spotipy
from matplotlib import style
from spotipy import util
from spotipy.oauth2 import SpotifyClientCredentials 



from pandas import json_normalize

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
############# Getting an access token ###############
############# A executer à chaque fois ###############
# My ID's
CLIENT_ID = '0dcb28d2945241c481bc7023b0b7d6e1'
CLIENT_SECRET = '4831fa5bf30b4a6a98804bbb90f72274'

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


############# A partir d'une playlist trouver les id des sons ###############

# A partir de l'ID d'une playlist trouver les sons qui sont dedans
def get_songs(playlist_id):
    r = requests.get(BASE_URL + 'playlists/' + playlist_id, headers=headers)
    r = r.json()
    for songs in r['tracks']:
        

r = json_normalize(r)

playlist_id = '2P0IaO6e5pU5IjnVsZH6Z2'
r = requests.get(BASE_URL + 'playlists/' + playlist_id, headers=headers)
    
playlist_link = "https://open.spotify.com/playlist/37i9dQZEVXbNG2KDcFcKOF?si=1333723a6eff4b7f"

def get_features_from_playlist(playlist_link) :
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]
    
    col_names =  ["track_uri", "track_name", "artist_uri","artist_info","artist_name","artist_pop","artist_genres","album",'track_pop','acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
    df  = pd.DataFrame(columns = col_names)

    for track in sp.playlist_tracks(playlist_URI)["items"]:
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

data = get_features_from_playlist(playlist_link)   

playlist_link = "https://open.spotify.com/playlist/13bvOZl3gV7si5jRHpBzeI"
data = get_features_from_playlist(playlist_link)

playlist_link2 = "https://open.spotify.com/playlist/67P7kU1wEfx1oCQUmsVWQu"
data2 = get_features_from_playlist(playlist_link2)

############# Getting information on a song ###############
def get_info(track_id):
    # actual GET request with proper header
    r = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    r = r.json()
    r = json_normalize(r)
    r = 
    return r

info = get_info("11dFghVXANMlKmJXsNCbNl")
info = json_normalize(info)
# Example
# Track ID from the URI
track_id = '6y0igZArWVi6Iz0rj35c1Y'
get_info(track_id)

artist_id = '2P0IaO6e5pU5IjnVsZH6Z2'
r = requests.get(BASE_URL + 'playlists/' + artist_id , headers=headers, params={'include_groups': 'album', 'limit': 50}, fields=tracks.items(added_at,added_by.id))
d = r.json()
def get_album(artist_id): 
    r = requests.get(BASE_URL + 'artists/' + artist_id + '/albums', headers=headers, params={'include_groups': 'album', 'limit': 50})
    d = r.json()
    
    data = []   # will hold all track info
    albums = [] # to keep track of duplicates
    
    # loop over albums and get all tracks
    for album in d['items']:
        album_name = album['name']
        
        # here's a hacky way to skip over albums we've already grabbed
        trim_name = album_name.split('(')[0].strip()
        if trim_name.upper() in albums or int(album['release_date'][:4]) > 1983:
            continue
        albums.append(trim_name.upper()) # use upper() to standardize
        
        # this takes a few seconds so let's keep track of progress    
        print(album_name)
        
        # pull all tracks from this album
        r = requests.get(BASE_URL + 'albums/' + album['id'] + '/tracks', headers=headers)
        tracks = r.json()['items']
    
        for track in tracks:
            # get audio features (key, liveness, danceability, ...)
            f = requests.get(BASE_URL + 'audio-features/' + track['id'], headers=headers)
            f = f.json()
        
            # combine with album info
            f.update({
                'track_name': track['name'],
                'album_name': album_name,
                'short_album_name': trim_name,
                'release_date': album['release_date'],
                'album_id': album['id']
                })
        
            data.append(f) 
    df = pd.DataFrame(data)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df.sort_values(by='release_date')
    # Zeppelin-specific: get rid of live album, remixes, vocal tracks, ...
    df = df.query('short_album_name != "The Song Remains The Same"')
    df = df[~df['track_name'].str.contains('Live|Mix|Track')]
    
    return data

# Example : donne l'album d'un artiste et analyse tous ses sons
artist_id = '36QJpDe2go2KgaRleHCDTp'
data = get_album(artist_id)

############# Obtenir les features à partir d'une playlist spotify ###############

def user_playlist_tracks_full(spotify_connection, user, playlist_id=None, fields=None, market=None):
    """ Get full details of the tracks of a playlist owned by a user.
        https://developer.spotify.com/documentation/web-api/reference/playlists/get-playlists-tracks/

        Parameters:
            - user - the id of the user
            - playlist_id - the id of the playlist
            - fields - which fields to return
            - market - an ISO 3166-1 alpha-2 country code.
    """

    # first run through also retrieves total no of songs in library
    response = spotify_connection.user_playlist_tracks(user, playlist_id, fields=fields, limit=100, market=market)
    results = response["items"]

    # subsequently runs until it hits the user-defined limit or has read all songs in the library
    while len(results) < response["total"]:
        response = spotify_connection.user_playlist_tracks(
            user, playlist_id, fields=fields, limit=100, offset=len(results), market=market
        )
        results.extend(response["items"])

user_playlist_tracks_full(spotify, '0dcb28d2945241c481bc7023b0b7d6e1', playlist_id='Thomas_aujoux', fields=None, market=None)

user = '0dcb28d2945241c481bc7023b0b7d6e1'
playlist_id='Thomas_aujoux'

sp = spotipy.Spotify()
def get_features_for_playlist(uri):
    playlist_id = uri.split(':')[2]
    results = sp.user_playlist(username, playlist_id)
    
get_features_for_playlist('spotify:playlist:2P0IaO6e5pU5IjnVsZH6Z2')
    
get_features_for_playlist(uri)
sp.user_playlist('21a2j53ksave2oldrdd54l4fq', '2P0IaO6e5pU5IjnVsZH6Z2')
    
############# Première analyse des sons ###############

# Répartition des sons d'un artiste en fonction des caractéristiques valence et acousticness
plt.figure(figsize=(10,10))

ax = sns.scatterplot(data=data, x='valence', y='acousticness', 
                     hue='short_album_name', palette='rainbow', 
                     size='duration_ms', sizes=(50,1000), 
                     alpha=0.7)

# display legend without `size` attribute
h,labs = ax.get_legend_handles_labels()
ax.legend(h[1:10], labs[1:10], loc='best', title=None)
