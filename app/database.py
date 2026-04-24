import pickle

def load_database(path="data/embeddings.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)