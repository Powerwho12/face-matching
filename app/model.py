from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def get_embedding(image):
    face = mtcnn(image)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face)

    return embedding.squeeze(0)