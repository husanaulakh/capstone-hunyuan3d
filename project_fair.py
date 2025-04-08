#!/usr/bin/env python3
"""
project_fair.py

A standalone script for Hunyuan3D-2 that takes a single input image,
runs it through the shape and texture pipelines, and exports a textured
3D mesh as a .glb file into a folder called "Project Fair".

Usage:
    python project_fair.py --image path/to/input.jpg [--output filename.glb] [--device cuda]

Dependencies:
    - torch, torchvision
    - Pillow
    - trimesh
    - hy3dgen (from the Hunyuan3D-2 repository)
"""

import os
import argparse
import torch
from PIL import Image
import trimesh

# Import the pipelines from hy3dgen package.
# These modules should be available since you cloned the Hunyuan3D-2 repo.
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def load_image(image_path):
    """Load the input image and convert it to RGB."""
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image_path}")
        return image
    except Exception as e:
        print("Error loading image:", e)
        raise

def generate_bare_mesh(image, device="cuda"):
    """Run the shape generation pipeline on the input image to produce a bare mesh."""
    print("Loading shape generation pipeline (Hunyuan3DDiTFlowMatchingPipeline)...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",  # Adjust this identifier if you wish to use a different model.
        device=device
    )
    print("Running shape generation...")
    # The pipeline is expected to return a list of meshes; we take the first one.
    result = shape_pipeline(image=image)
    bare_mesh = result[0]
    print("Bare mesh generated.")
    return bare_mesh

def apply_texture(bare_mesh, image, device="cuda"):
    """Run the texture synthesis pipeline on the bare mesh using the input image."""
    print("Loading texture synthesis pipeline (Hunyuan3DPaintPipeline)...")
    tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",  # Adjust if using separate checkpoints.
        device=device
    )
    print("Running texture synthesis...")
    textured_mesh = tex_pipeline(bare_mesh, image=image)
    print("Texturing complete.")
    return textured_mesh

def save_mesh_glb(mesh, output_path):
    """Export the mesh (with texture) as a .glb file."""
    try:
        mesh.export(output_path)
        print(f"Mesh successfully saved to: {output_path}")
    except Exception as e:
        print("Error saving mesh:", e)
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2 Inference: Generate a textured .glb 3D model from an input image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default="output.glb", help="Output filename (with .glb extension).")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference ('cuda' or 'cpu').")
    args = parser.parse_args()

    # Load the input image.
    input_image = load_image(args.image)

    # Generate the bare mesh.
    bare_mesh = generate_bare_mesh(input_image, device=args.device)

    # Apply texture to the generated mesh.
    final_mesh = apply_texture(bare_mesh, input_image, device=args.device)

    # Ensure the output directory "Project Fair" exists.
    output_dir = os.path.join(os.getcwd(), "Project Fair")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Construct the full output file path.
    output_file = os.path.join(output_dir, args.output)

    # Save the final mesh as a .glb file.
    save_mesh_glb(final_mesh, output_file)

if __name__ == "__main__":
    main()