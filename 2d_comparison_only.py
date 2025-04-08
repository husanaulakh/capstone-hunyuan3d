import os
import torch
from PIL import Image
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import gradio as gr
from google.cloud import storage
import io
import re

# ---- CONFIG ---- #
model_path = 'checkpoint_capstone_whole_snowflake_42k.ptc'  # change to trained model
embedding_save_path = 'backdrop_embeddings_front.pt'

# ---- DEVICE ---- #
device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval()
print("Using Device:", device)

# ---- PREPROCESS ---- #
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- SIMILARITY & EMBEDDING UTILS ---- #
def compute_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze()
    return embedding

def find_top1_transparency_match(query_image_path):
    """
    • Computes the cosine similarities between the query image embedding and all backdrop embeddings.
    • Sorts the matches from highest to lowest similarity.
    • Iterates over the sorted indices and returns the first match whose filename starts with
      "random_texture_transparency_".
    • If none is found, returns the top match.
    """
    if not os.path.exists(embedding_save_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_save_path}")
    saved = torch.load(embedding_save_path)
    backdrop_embeddings = saved['embeddings']
    backdrop_names = saved['names']
    query_embedding = compute_embedding(query_image_path)
    
    # Compute cosine similarities with all embeddings
    sims = cosine_similarity(query_embedding.unsqueeze(0), torch.stack(backdrop_embeddings)).squeeze()
    
    # Sort indices in descending order by similarity
    sorted_indices = torch.argsort(sims, descending=True)
    
    transparency_prefix = "random_scaling/"
    # Iterate through sorted indices to find transparency augmentation
    for idx in sorted_indices:
        name = backdrop_names[idx.item()]
        # if name.startswith(transparency_prefix):
        best_match_name = name
        best_score = sims[idx].item()
        return best_match_name, best_score
    # If transparency augmentation not found, return top match
    top_index = sorted_indices[0].item()
    return backdrop_names[top_index], sims[top_index].item()

# ---- HELPER FUNCTION: CONSTRUCT GCS FILEPATH ---- #
def construct_file_path(filename: str) -> str:
    """
    Constructs a GCS file path based on filename structure.
    
    Augmentation files:
      • Expected format: "<aug_prefix>_<contract>-<asset>-<view>.png"
      • Example Input: "morph_object_0x0111ac7e9425c891f935c4ce54cf16db7c14b7db-1023-1_front.png"
      • Parsed:
          - aug_prefix = "morph_object"
          - final_filename = "0x0111ac7e9425c891f935c4ce54cf16db7c14b7db-1023-1_front.png"
          - contract = "0x0111ac7e9425c891f935c4ce54cf16db7c14b7db"
          - asset = "0x0111ac7e9425c891f935c4ce54cf16db7c14b7db-1023"
      • Final GCS path:
          yakoa-3d-augmentations/3d-assets-augmentation-with-docker/
            {contract}/{asset}/{aug_prefix}/{final_filename}
    
    Non-augmentation files:
      • Expected format: "<contract>-<asset>-<view>.png"
      • Example Input: "0x00fda66072ac818db214e0cf7302c0f458202b5d-11-1_front.png"
      • Parsed:
          - contract = "0x00fda66072ac818db214e0cf7302c0f458202b5d"
          - asset = "0x00fda66072ac818db214e0cf7302c0f458202b5d-11"
      • Final GCS path:
          yakoa-nft-catalog/2D-representations/
            {contract}/{asset}/{filename}
    """
    # List of augmentation prefixes
    augmentations = [
        "add_noise_jitter", "change_material", "morph_object", "original_asset",
        "random_bending", "random_color_shift", "random_non_uniform_scaling",
        "random_rotate", "random_scaling", "random_texture_transparency",
        "random_twist", "shear_object"
    ]
    
    for aug in augmentations:
        if filename.startswith(aug + "_"):
            # Augmentation file detected.
            # Remove the augmentation prefix and underscore for the final filename.
            final_filename = filename[len(aug)+1:]
            # Split the final filename to extract contract and asset.
            parts = final_filename.split("-", 2)
            if len(parts) < 3:
                raise ValueError("Invalid augmentation filename format.")
            contract = parts[0]
            asset = f"{contract}-{parts[1]}"
            return (f"yakoa-3d-augmentations/3d-assets-augmentation-with-docker/"
                    f"{contract}/{asset}/{aug}/{final_filename}")
    
    # Non-augmentation file assumed format: "<contract>-<asset>-<view>.png"
    parts = filename.split("-", 2)
    if len(parts) < 3:
        raise ValueError("Invalid non-augmentation filename format.")
    contract = parts[0]
    asset = f"{contract}-{parts[1]}"
    return (f"yakoa-nft-catalog/2D-representations/{contract}/{asset}/{filename}")

def preprocess_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image to speed up processing."""
    image.thumbnail((max_size, max_size))
    return image

from io import BytesIO
import gradio as gr
from rembg import remove


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using rembg and return RGBA PIL image."""
    if image is None:
        raise ValueError("No image provided.")

    image = preprocess_image(image)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    input_bytes = buffer.getvalue()

    output_bytes = remove(input_bytes)
    output_image = Image.open(BytesIO(output_bytes)).convert("RGBA")
    return output_image

# ---- HELPER FUNCTION: FETCH IMAGE FROM GCS ---- #
def fetch_image_from_gcs(gcs_path: str) -> Image.Image:
    """
    Expects a GCS path in the form "bucket_name/path/to/blob".
    Downloads and returns the image.
    """
    parts = gcs_path.split('/', 1)
    if len(parts) != 2:
        raise ValueError("Invalid GCS path format. Expected 'bucket_name/path/to/blob'.")
    bucket_name, blob_name = parts
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = remove_background(image)
    return image


# ---- NEW FUNCTION: FIND TOP 10 MATCHES ---- #
def find_top10_matches(query_image_path):
    if not os.path.exists(embedding_save_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_save_path}")
    saved = torch.load(embedding_save_path)
    backdrop_embeddings = saved['embeddings']
    backdrop_names = saved['names']
    query_embedding = compute_embedding(query_image_path)
    
    # Compute cosine similarities with all embeddings
    sims = cosine_similarity(query_embedding.unsqueeze(0), torch.stack(backdrop_embeddings)).squeeze()
    
    # Sort indices in descending order by similarity and get top 10
    sorted_indices = torch.argsort(sims, descending=True)
    top10_indices = sorted_indices[:10]
    
    top10_names = [backdrop_names[idx.item()] for idx in top10_indices]
    top10_scores = [sims[idx].item() for idx in top10_indices]
    return top10_names, top10_scores

# ---- UPDATED GRADIO INTERFACE FUNCTION ---- #
def gradio_predict(image):
    # Save the uploaded image temporarily
    temp_path = "temp_query_img.jpg"
    im = Image.fromarray(image)
    im.save(temp_path)
    
    # Get top 10 matches
    try:
        top10_names, top10_scores = find_top10_matches(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return None, f"Error computing matches: {str(e)}"
    
    # Lists to store matched images and details
    matched_images = []
    details = []
    
    for name, score in zip(top10_names, top10_scores):
        try:
            gcs_path = construct_file_path(name)
            img = fetch_image_from_gcs(gcs_path)
            matched_images.append(img)
            details.append(f"Name: {name}\nScore: {score:.4f}\nGCS: {gcs_path}")
        except Exception as e:
            # Skip image if error occurs fetching from GCS
            details.append(f"Name: {name} -- Error fetching image: {str(e)}")
    
    os.remove(temp_path)
    return matched_images, "\n\n".join(details)

# ---- DEFINE AND LAUNCH GRADIO INTERFACE WITH GALLERY OUTPUT ---- #
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy", label="Upload Query Image"),
    outputs=[gr.Gallery(label="Top 10 Matched Images", show_label=True), 
             gr.Textbox(label="Match Details")],
    title="Image Similarity Finder with GCS (Top 10)",
    description="Upload an image to find the top 10 transparency augmentation matches and display the associated images from Google Cloud Storage."
)

if __name__ == "__main__":
    iface.launch()

# ---- GRADIO INTERFACE FUNCTION ---- #
# def gradio_predict(image):
#     # Save the uploaded image temporarily
#     temp_path = "temp_query_img.jpg"
#     im = Image.fromarray(image)
#     im.save(temp_path)
    
#     # Compute best match using transparency iteration
#     match_name, score = find_top1_transparency_match(temp_path)
    
#     # Construct GCS file path based on the matched filename
#     gcs_path = construct_file_path(match_name)
    
#     # Fetch the matched image from Google Cloud Storage
#     try:
#         matched_image = fetch_image_from_gcs(gcs_path)
#     except Exception as e:
#         os.remove(temp_path)
#         return None, f"Error fetching image from GCS: {str(e)}"
    
#     os.remove(temp_path)
    
#     # Format match details
#     result_text = f"Match Name: {match_name}\nScore: {score:.4f}\nGCS Path: {gcs_path}"
#     return matched_image, result_text

# ---- DEFINE AND LAUNCH GRADIO INTERFACE ---- #
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy", label="Upload Query Image"),
    outputs=[gr.Image(label="Matched Image"), gr.Textbox(label="Match Details")],
    title="Image Similarity Finder with GCS",
    description="Upload an image to find the transparency augmentation match and display the associated image from Google Cloud Storage."
)

if __name__ == "__main__":
    iface.launch()
