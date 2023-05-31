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





############# Getting information on a song ###############
def get_info(track_id):
    # actual GET request with proper header
    r = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    r = r.json()
    return r

# Example
# Track ID from the URI
track_id = '6y0igZArWVi6Iz0rj35c1Y'
get_info(track_id)

    
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


    
