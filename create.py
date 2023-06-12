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
cid = '92278466ab914b5a8cd997ef7ad26bf0'
secret = '42a4144330f2469e98bb6e50bb276cb5'
redirect_uri='http://localhost:7777/callback'
username = '31xl6hkcysxjzwhu37gl7prssx7i'




######### Authorization flow #########
scope = "playlist-modify-public"
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)



######### Spotipy manipulations #########

def create_playlist(sp, username, playlist_name, playlist_description, playlist_tracks):
    playlists = sp.user_playlist_create(username, playlist_name, description = playlist_description)
    playlist_id = fetch_playlists(sp,username)["id"][0]
    index = 0
    results = []
    while index < len(playlist_tracks):
        results += sp.user_playlist_add_tracks(username, playlist_id, tracks = playlist_tracks[index:index + 100])
        index += 100


# Exemple 
sp = sp
username = username
playlist_name = 'Pandas Party'
playlist_description = 'A pure party playlist created by DJ Pandas!'
playlist_tracks = df_party_exp_I.index
create_playlist(sp, username, playlist_name, playlist_description, playlist_tracks)




def enrich_playlist(sp, username, playlist_id, playlist_tracks):
    index = 0
    results = []
    
    while index < len(playlist_tracks):
        results += sp.user_playlist_add_tracks(username, playlist_id, tracks = playlist_tracks[index:index + 100])
        index += 100
        index += 100


sp = sp
username = username
playlist_id = playlist_id
playlist_tracks = df_party_exp_I.index
enrich_playlist(sp, username, '779Uv1K6LcYiiWxblSDjx7', list_track)






