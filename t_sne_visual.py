import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from sklearn import preprocessing

df = pd.read_csv("outputs/full_dataset_features.csv")
counts = df['genre'].value_counts()
masked = df[df['genre'].isin(["classic rock", "alternative metal"])]
x = masked.iloc[:, :-4].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_scaled = pd.DataFrame(x_scaled)

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(df_scaled)

obj_tsne_df = pd.DataFrame()
obj_tsne_df['tsne-2d-one'] = tsne_results[:, 0]
obj_tsne_df['tsne-2d-two'] = tsne_results[:, 1]
obj_tsne_df["track_name"] = masked["track_name"].tolist()
obj_tsne_df["album_name"] = masked["album_name"].tolist()
obj_tsne_df["artist_name"] = masked["artist_name"].tolist()
obj_tsne_df["genre"] = masked["genre"].tolist()

t_SNE_fig_doc = px.scatter(obj_tsne_df, x='tsne-2d-one', y='tsne-2d-two', color="genre", hover_data=["track_name"],
                           color_discrete_sequence=px.colors.qualitative.Dark24)

t_SNE_fig_doc.update_traces(marker=dict(size=17, opacity=0.8), selector=dict(mode='markers'))

t_SNE_fig_doc.write_html("t_SNE_comparison2.html")


data = df.iloc[:, :40].values
m, k = data.shape

mat = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        if i != j:
            mat[i][j] = cosine(data[i,:], data[j,:])
        else:
            mat[i][j] = 0.

distance_mat = pd.DataFrame(mat)
