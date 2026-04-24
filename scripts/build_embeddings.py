import os
import pickle
from PIL import Image
import torch

from app.model import get_embedding

DATA_DIR = "data/celebrities"
OUTPUT = "data/embeddings.pkl"

def build():
    db = {}

    for celeb_name in os.listdir(DATA_DIR):
        celeb_path = os.path.join(DATA_DIR, celeb_name)


        if not os.path.isdir(celeb_path):
            continue

        embeddings = []

        for file in os.listdir(celeb_path):
            img_path = os.path.join(celeb_path, file)


            if not os.path.isfile(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                emb = get_embedding(img)

                if emb is not None:
                    embeddings.append(emb)
            except:
                continue

        if embeddings:
            avg_emb = torch.mean(torch.stack(embeddings), dim=0)
            db[celeb_name] = avg_emb

            print(f"{celeb_name} done")

    with open(OUTPUT, "wb") as f:
        pickle.dump(db, f)

    print("DB saved!")

if __name__ == "__main__":
    build()