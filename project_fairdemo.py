import os
import torch
from PIL import Image
from io import BytesIO
import gradio as gr
from rembg import remove
import trimesh

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


# Load model once at startup (not inside the function)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ Loading Hunyuan3D pipeline once on startup... [{device}]")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2", device=device)
print("✅ Model loaded.")


def preprocess_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image to speed up processing."""
    image.thumbnail((max_size, max_size))
    return image


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


def generate_3d_shape(image: Image.Image) -> tuple:
    """
    Generate GLB mesh from background-removed image.
    Returns: (path to .glb for preview and download)
    """
    if image is None:
        raise ValueError("No image provided.")

    print("🔪 Removing background...")
    bg_removed_image = remove_background(image)

    print("🧠 Generating 3D shape...")
    result = pipeline(image=bg_removed_image)
    mesh = result[0]

    if isinstance(mesh, trimesh.Trimesh):
        mesh.remove_unreferenced_vertices()
        mesh.fill_holes()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()

    output_dir = "Project Fair"
    os.makedirs(output_dir, exist_ok=True)

    glb_path = os.path.join(output_dir, "output_shape.glb")
    mesh.export(glb_path)
    print(f"✅ Mesh saved to: {glb_path}")
    return glb_path, glb_path


# Gradio UI
with gr.Blocks(title="⚡ Fast 3D Mesh Generator") as demo:
    gr.Markdown("## 🧠 Image → Background Removed → 3D GLB Mesh")
    gr.Markdown("Upload an image, preview background removal, generate a 3D mesh, and download it.")

    input_image = gr.Image(type="pil", label="📤 Upload Image")

    with gr.Row():
        bg_preview = gr.Image(type="pil", label="🎯 Background Removed Preview")
        process_btn = gr.Button("🚀 Generate 3D Mesh")

    with gr.Row():
        model_preview = gr.Model3D(label="🔄 3D Preview (.glb)")
        model_download = gr.File(label="⬇️ Download GLB")

    # Step 1: Background removal preview (resized internally)
    input_image.change(fn=remove_background, inputs=input_image, outputs=bg_preview)

    # Step 2: Mesh generation
    process_btn.click(fn=generate_3d_shape, inputs=input_image, outputs=[model_preview, model_download])


if __name__ == "__main__":
    demo.launch()
