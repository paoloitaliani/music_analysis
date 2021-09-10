import telebot
import os
import linecache
import sys

TOKEN = os.getenv("TOKEN")
bot = telebot.TeleBot(TOKEN)


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    out = 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
    return out


@bot.message_handler(commands=["try"])
def trye(message):
    bot.send_message(message.chat.id, "hello hello")

    try:
        from spotipy.oauth2 import SpotifyClientCredentials
        from sklearn import preprocessing
        from urllib.parse import unquote
        from pydub import AudioSegment
        from bs4 import BeautifulSoup
        from annoy import AnnoyIndex
        import pandas as pd
        import numpy as np
        import requests
        import spotipy
        import librosa
        import json

        def download_mp3(track_id):
            link = "https://open.spotify.com/embed/track/{}".format(track_id)
            req = requests.get(link)
            soup = BeautifulSoup(req.content.decode('utf-8', 'ignore'), 'html.parser')
            supa = soup.find('script', attrs={'id': 'resource'})
            string = supa.contents[0].strip()
            url = unquote(string)

            preview_url = json.loads(url)["preview_url"]

            r = requests.get(preview_url)

            output = os.path.join("data", "mp3_preview", "song.mp3")
            if not os.path.exists(os.path.dirname(output)):
                try:
                    os.makedirs(os.path.dirname(output))
                except:
                    pass

            with open(output, "wb") as output:
                output.write(r.content)

        def getTrackFeatures(track_id):
            meta = sp.track(track_id)
            features = sp.audio_features(track_id)

            # meta
            name = meta['name']
            album = meta['album']['name']
            artist = meta['album']['artists'][0]['name']

            # spotify features
            acousticness = features[0]['acousticness']
            danceability = features[0]['danceability']
            energy = features[0]['energy']
            instrumentalness = features[0]['instrumentalness']
            liveness = features[0]['liveness']
            loudness = features[0]['loudness']
            speechiness = features[0]['speechiness']
            tempo = features[0]['tempo']
            time_signature = features[0]['time_signature']

            # wav features
            download_mp3(track_id)
            root = os.path.join("data", "mp3_preview", "song.mp3")
            sound = AudioSegment.from_mp3(root)
            output = os.path.join("data", "wav_previews", "song.wav")
            if not os.path.exists(os.path.dirname(output)):
                try:
                    os.makedirs(os.path.dirname(output))
                except:
                    pass

            with open(output, "wb") as f:
                sound.export(f, format="wav")

            # naming variables
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
            x, sr = librosa.load(output, sr=44100)

            row = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40), axis=1).tolist()
            row.extend(np.mean(librosa.feature.chroma_stft(y=x, sr=sr), axis=1).tolist())
            row.append(np.mean(librosa.feature.spectral_centroid(y=x, sr=sr)))
            row.append(np.mean(librosa.feature.spectral_bandwidth(y=x, sr=sr)))
            row.append(np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr)))
            row.append(np.mean(librosa.feature.zero_crossing_rate(x)))

            row.append(acousticness)
            row.append(danceability)
            row.append(energy)
            row.append(instrumentalness)
            row.append(liveness)
            row.append(loudness)
            row.append(speechiness)
            row.append(tempo)
            row.append(time_signature)

            row.append(name)
            row.append(album)
            row.append(artist)

            return row

        class IndexClass:
            def __init__(self, vectors, labels):
                self.dimension = vectors.shape[1]
                self.vectors = vectors.astype('float32')
                self.labels = labels

            def build(self, number_of_trees=5):
                self.index = AnnoyIndex(self.dimension)
                for i, vec in enumerate(self.vectors):
                    self.index.add_item(i, vec.tolist())
                self.index.build(number_of_trees)

            def query(self, vector, k=10):
                indices = self.index.get_nns_by_vector(
                    vector,
                    k
                )
                return [self.labels[i] for i in indices]

        os.environ["SPOTIPY_CLIENT_ID"] = os.getenv("CLIENT_ID")
        os.environ["SPOTIPY_CLIENT_SECRET"] = os.getenv("CLIENT_SECRET")
        # TOKEN = os.getenv("TOKEN")
        client_credentials_manager = SpotifyClientCredentials()
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        # bot = telebot.TeleBot(TOKEN)

        bot.send_message(message.chat.id, "ok1")

        path = os.path.join("outputs", "full_dataset_features.csv")
        df = pd.read_csv(path)
        x = df.iloc[:, :-4].values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_scaled = pd.DataFrame(x_scaled)
        df_numpy = df_scaled.to_numpy()

        artist = "beatles"
        track = "yesterday"
        track_id = sp.search(q='artist:' + artist + ' track:' + track, type='track')
        row = getTrackFeatures(track_id["tracks"]["items"][0]["id"])[:-3]
        reshaped_row = np.asarray(row).reshape(1, -1)
        scaled_row = min_max_scaler.transform(reshaped_row)
        scaled_row = scaled_row.tolist()

        index = IndexClass(df_numpy, df["track_name"])
        index.build()
        recommendations = index.query(scaled_row[0])

        bot.send_message(message.chat.id, recommendations[0])

    except:
        bot.send_message(message.chat.id, PrintException())


bot.polling()
