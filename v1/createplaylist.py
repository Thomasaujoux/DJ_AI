

######### Librairies #########
# Pour gérer le système
import os

# Pour la connexion avec Spotify
import spotipy
import spotipy.util as util
from spotifyclient import SpotifyClient

# Pour le script
import numpy as np



def main():


    ######### Declare the credentials #########
    # Propre à notre application
    cid = '86696316a6b446188061b1dbe24eab70'
    secret = '1cca02c822f84043b9875cf8568d2f89'
    # Pour tester sur en local
    redirect_uri='http://localhost:7777/callback'
    # Pour les autorisations concernant l'utilisateur
    username = '21a2j53ksave2oldrdd54l4fq'


    ######### Authorization flow #########
    # Les différentes autorisations concernant l'application Spotify
    scope = "playlist-modify-public playlist-modify-private user-read-recently-played playlist-read-private playlist-read-collaborative"
    # Comme l'application n'est pas déclarée auprès de Spotify le Token dure juste 1h, il faut le recharger à chaque fois
    token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)
    # Il est possible qu'on atteigne le nombre maximal de requêtes à un moment, il faudrat récréer une application sur le Dashboard

    # Génération du Token
    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)
    spotify_client = SpotifyClient(token, username)



    ######### Begining of the script #########
    # get all the playlists from an user
    playlists = spotify_client.get_user_playlists(username)
    print("\nHere are all the playlists you have in your Spotify account: ")
    for index, playlist in enumerate(playlists):
        print(f"{index+1}- Le nom de la playlist est '{playlist.name}' - l'Id à recopier par la suite est: {playlist.id.split(':')[-1]}")
    
    
    # choose which playlist to use as a seed to generate a new playlist
    index_playlist = input("\nEnter the id of the playlist you want: ")


    # get the seeds of the tracks you want to put in the recommendation
    tracks = spotify_client.get_info_playlist(index_playlist)
    for index, track in enumerate(tracks):
        print(f"{index+1}- {track}")

    
    # Create seed tracks which will be used for the recommendation
    seed_tracks = [track.id for track in tracks]
    length = len(seed_tracks)
    num_tracks_to_visualise = int(input("How many tracks would you like to have in your playlist ? "))
    vect = np.linspace(0, length + num_tracks_to_visualise, length).astype(int)
    print(type(tracks))
    # get recommended tracks based off seed tracks
    recommended_tracks = tracks[0]
 #   for i in range(len(vect)):
 #       num_tracks_to_recommend = int((vect[i] + vect[i+1])/2)-1
#        print(seed_tracks)
 #       print(seed_tracks[i])
 #       print(num_tracks_to_recommend)
 #       print(recommended_tracks)
 #       print(spotify_client.get_track_recommendations(seed_tracks[i], limit=num_tracks_to_recommend))
 #       recommended_tracks = spotify_client.get_track_recommendations(seed_tracks[i], limit=num_tracks_to_recommend)
  #      recommended_tracks = tracks[i]
    print("\nHere are the recommended tracks which will be included in your new playlist:")
    print(str(seed_tracks[0]))
    a = spotify_client.get_track_recommendations(str(seed_tracks[0]), limit=2)
    print(a)
    for index, track in enumerate(recommended_tracks):
        print(f"{index+1}- {track}")

    # get playlist name from user and create playlist
    playlist_name = input("\nWhat's the playlist name? ")
    playlist = spotify_client.create_playlist(playlist_name)
    print(f"\nPlaylist '{playlist.name}' was created successfully.")

    # populate playlist with recommended tracks
    spotify_client.populate_playlist(playlist, recommended_tracks)
    print(f"\nRecommended tracks successfully uploaded to playlist '{playlist.name}'.")


if __name__ == "__main__":
    main()