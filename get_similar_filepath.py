import os
import torch
import re
from PIL import Image
from torchvision import transforms
from torch.nn.functional import cosine_similarity

# ---- CONFIG ---- #
model_path = 'checkpoint_incandescent_rooster_105k.ptc' #change to trained model
embedding_save_path = 'backdrop_embeddings_front.pt'
backdrop_dir = './test_flatten'

# ---- DEVICE ---- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval()
print("Using Device:", device)

# ---- PREPROCESS ---- #
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- UTILITIES ---- #
def compute_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze()
    return embedding

def find_top1_match(query_image_path):
    if not os.path.exists(embedding_save_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_save_path}")

    saved = torch.load(embedding_save_path)
    backdrop_embeddings = saved['embeddings']
    backdrop_names = saved['names']

    query_embedding = compute_embedding(query_image_path)
    sims = cosine_similarity(query_embedding.unsqueeze(0), torch.stack(backdrop_embeddings)).squeeze()
    best_index = torch.argmax(sims).item()

    best_match_name = backdrop_names[best_index]
    best_score = sims[best_index].item()

    return best_match_name, best_score

# ---- USAGE ---- #
if __name__ == "__main__":
    query_img = "" # Change this path
    match_name, score = find_top1_match(query_img)
    print(f"Top match for query image:\n→ {query_img}\n\nBest match:\n→ {match_name} (score={score:.4f})")