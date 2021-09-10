import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import requests
import os
import json
from urllib.parse import unquote
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm
import numpy as np
import librosa

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_artist_albums(artist_name="", CLIENT_ID="", CLIENT_SECRET="", threshold_year=2100):
    # get artist id
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    result = sp.search(artist_name)
    artist_id = result['tracks']['items'][0]['artists'][0]["id"]

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

    headers = {
        'Authorization': 'Bearer {token}'.format(token=access_token)
    }

    BASE_URL = 'https://api.spotify.com/v1/'

    # pull all artists albums
    r = requests.get(BASE_URL + 'artists/' + artist_id + '/albums',
                     headers=headers,
                     params={'include_groups': 'album', 'limit': 50})
    d = r.json()

    albums = []  # to keep track of duplicates
    tracks_list = []  # to keep track of tracks duplicates
    for album in d['items']:

        try:
            album_name = album['name']

            # here's a hacky way to skip over albums we've already grabbed
            trim_name = album_name.split('(')[0].strip()
            trim_name = trim_name.split('[')[0].strip()
            if trim_name.upper() in albums or int(album['release_date'][:4]) > threshold_year:
                continue
            albums.append(trim_name.upper())  # use upper() to standardize

            # this takes a few seconds so let's keep track of progress
            print(album_name)

            # pull all tracks from this album
            r = requests.get(BASE_URL + 'albums/' + album['id'] + '/tracks',
                             headers=headers)
            tracks = r.json()['items']

            track_numb = 0
            for track in tracks:

                track_numb += 1

                try:

                    track_id = track["id"]
                    track_name = track["name"]

                    trim_name = track_name.split('-')[0].strip()
                    if trim_name.upper() in tracks_list:
                        continue

                    tracks_list.append(trim_name.upper())

                    link = "https://open.spotify.com/embed/track/{}".format(track_id)
                    req = requests.get(link)
                    soup = BeautifulSoup(req.content.decode('utf-8', 'ignore'), 'html.parser')
                    supa = soup.find('script', attrs={'id': 'resource'})
                    string = supa.contents[0].strip()
                    url = unquote(string)

                    preview_url = json.loads(url)["preview_url"]

                    r = requests.get(preview_url)

                    output = "data/mp3_previews/{}/{}-{}.mp3".format(artist_name, artist_name, track_name)
                    if not os.path.exists(os.path.dirname(output)):
                        try:
                            os.makedirs(os.path.dirname(output))
                        except:
                            pass

                    with open(output, "wb") as output:
                        output.write(r.content)

                    # create json

                    json_output = {}
                    json_output["track_name"] = track_name
                    json_output["album_name"] = album_name
                    json_output["artist_name"] = artist_name
                    json_output["path"] = "data/wav_previews/{}/{}-{}.wav".format(artist_name, artist_name, track_name)

                    output = "data/json_outputs/{}/{}-{}.json".format(artist_name, artist_name, track_name)
                    if not os.path.exists(os.path.dirname(output)):
                        try:
                            os.makedirs(os.path.dirname(output))
                        except:
                            pass

                    with open(output, 'w') as outfile:
                        json.dump(json_output, outfile)

                except:
                    print("failed to extract data for track {} of album {}".format(track_numb, album_name))

        except:
            print("failed to extract data for album {}".format(album["name"]))


def from_mp3_to_wav(artist):
    root = "/Users/Niolo/Desktop/Github/music_analysis/data/mp3_previews/{}".format(artist)
    file_list = os.listdir(root)
    for file in tqdm(file_list):

        if 'mp3' in file:
            sound = AudioSegment.from_mp3("{}/{}".format(root, file))
            output = "data/wav_previews/{}/{}.wav".format(artist, file[:-4])
            if not os.path.exists(os.path.dirname(output)):
                try:
                    os.makedirs(os.path.dirname(output))
                except:
                    pass

            with open(output, "wb") as f:
                sound.export(f, format="wav")


def build_df(artist):
    root = "/Users/Niolo/Desktop/Github/music_analysis/data/json_outputs/{}".format(artist)
    dirlist = [item for item in os.listdir(root)]

    inner_data_list = []
    for file in dirlist:

        # If file is a json, construct it's full path and open it, append all json data to list
        if 'json' in file:
            json_path = os.path.join(root, file)
            with open(json_path) as json_file:
                data = json.load(json_file)
            inner_data_list.append(data)

    column_names = []
    for i in range(1, 41):
        column_names.append("mfcc_{}".format(i))

    for i in range(1, 13):
        column_names.append("chroma_{}".format(i))

    column_names.append("spec_cent")
    column_names.append("spec_bw")
    column_names.append("rolloff")
    column_names.append("zcr")
    column_names.append("track_name")
    column_names.append("album_name")
    column_names.append("artist_name")

    row_list = []
    for track in tqdm(inner_data_list):
        try:
            x, sr = librosa.load(track["path"], sr=44100)

            row = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40), axis=1).tolist()
            row.extend(np.mean(librosa.feature.chroma_stft(y=x, sr=sr), axis=1).tolist())
            row.append(np.mean(librosa.feature.spectral_centroid(y=x, sr=sr)))
            row.append(np.mean(librosa.feature.spectral_bandwidth(y=x, sr=sr)))
            row.append(np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr)))
            row.append(np.mean(librosa.feature.zero_crossing_rate(x)))

            row.append(track["track_name"])
            row.append(track["album_name"])
            row.append(track["artist_name"])
            row_list.append(row)

        except Exception as e:
            print(e)

    df = pd.DataFrame(row_list, columns=column_names)
    return df


def create_df_full_dataset(row_list):
    column_names = []

    for i in range(1, 41):
        column_names.append("mfcc_{}".format(i))

    for i in range(1, 13):
        column_names.append("chroma_{}".format(i))

    column_names.append("spec_cent")
    column_names.append("spec_bw")
    column_names.append("rolloff")
    column_names.append("zcr")

    column_names.append("acusticness")
    column_names.append("danceability")
    column_names.append("energy")
    column_names.append("instrumentalness")
    column_names.append("liveness")
    column_names.append("loudness")
    column_names.append("speechiness")
    column_names.append("tempo")
    column_names.append("time_signature")

    column_names.append("track_name")
    column_names.append("album_name")
    column_names.append("artist_name")

    df = pd.DataFrame(row_list, columns=column_names)

    return df


def get_all_data():

    playlists = sp.user_playlists('spotify')
    spotify_playlist_ids = []
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            spotify_playlist_ids.append(playlist['uri'][-22:])
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None

    def getTrackIDs(playlist_id):
        ids = []
        playlist = sp.user_playlist('spotify', playlist_id)
        for item in playlist['tracks']['items'][:10]:
            track = item['track']
            ids.append(track['id'])
        return ids

    row_list = []
    for playlist in tqdm(spotify_playlist_ids):
        try:
            tracks_ids = getTrackIDs(playlist)

            for track in tracks_ids:
                try:
                    row_list.append(getTrackFeatures(track))
                except Exception as e:
                    print(e)

        except Exception as e:
            print(e)

    return row_list


def get_artist_genre(df):
    genres_list = []
    for artist_name in tqdm(df["artist_name"]):
        try:
            result = sp.search(artist_name)
            track = result['tracks']['items'][0]
            artist = sp.artist(track["artists"][0]["external_urls"]["spotify"])
            genres_list.append(artist["genres"][0])
        except:
            genres_list.append("other")
            print(artist_name)

    df["genre"] = genres_list
    df.to_csv("outputs/full_dataset_features2.csv", index=False)


artist = "beatles"
track = "yesterday"
track_id = sp.search(q='artist:' + artist + ' track:' + track, type='track')
row = getTrackFeatures(track_id["tracks"]["items"][0]["id"])[:-3]
row = np.array(row)
scaled_row = min_max_scaler.transform(row)
scaled_row = scaled_row.tolist()