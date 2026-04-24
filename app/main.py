from fastapi import FastAPI, UploadFile
from PIL import Image
import io

from app.model import get_embedding
from app.database import load_database
from app.utils import find_top_k
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

celeb_db = load_database()


@app.get("/")
def root():
    return {"message": "Celebrity Lookalike API"}


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    emb = get_embedding(img)

    if emb is None:
        return {"error": "No face detected"}

    results = find_top_k(emb, celeb_db)

    return {
        "results": [
            {"name": name, "similarity": float(score)}
            for name, score in results
        ]
    }