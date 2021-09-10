from sklearn import preprocessing
from annoy import AnnoyIndex
import pandas as pd


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


if __name__ == "__main__":

    df = pd.read_csv("/Users/Niolo/Desktop/Github/music_analysis/outputs/full_dataset_features.csv")
    x = df.iloc[:, :-4].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_scaled = pd.DataFrame(x_scaled)
    df_numpy = df_scaled.to_numpy()
    index = IndexClass(df_numpy, df["track_name"])
    index.build()
    index.query(scaled_row[0])
