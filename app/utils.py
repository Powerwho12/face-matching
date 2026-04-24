import torch
import torch.nn.functional as F

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


def find_top_k(user_emb, celeb_db, k=5):
    results = []

    for name, emb in celeb_db.items():
        sim = cosine_similarity(user_emb, emb)
        results.append((name, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]