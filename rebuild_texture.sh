#!/bin/bash
# rebuild_texture.sh
# This script ensures that huggingface_hub is installed,
# then explicitly loads the Hunyuan3DPaintPipeline using the local checkpoint folder,
# so that you can verify that texture synthesis will work.

# 1. Ensure HuggingFace Hub is installed
echo "Installing huggingface_hub (if not already installed)..."
pip install --upgrade huggingface_hub

# 2. Define the expected texture checkpoint folder
# (Adjust the path if your checkpoint folder is located elsewhere.)
TEXTURE_MODEL_PATH="/home/husanaulakh5/.cache/huggingface/hub/models--tencent--Hunyuan3D-2/snapshots/efb40d890aeade24c68ec7555eb206de8ba68c77/hunyuan3d-paint-v2-0"

if [ -d "$TEXTURE_MODEL_PATH" ]; then
    echo "Found texture model folder: $TEXTURE_MODEL_PATH"
else
    echo "ERROR: Texture model folder 'hunyuan3d-paint-v2-0' not found at $TEXTURE_MODEL_PATH"
    echo "Please download the Hunyuan3D-Paint checkpoint from Hugging Face and place it here."
    exit 1
fi

# 3. Test loading the texture pipeline using Python.
echo "Testing Hunyuan3DPaintPipeline load using the texture checkpoint folder..."
python - <<EOF
from hy3dgen.texgen import Hunyuan3DPaintPipeline
print("Attempting to load texture pipeline from: '$TEXTURE_MODEL_PATH'")
# Load explicitly from the local checkpoint folder
pipeline = Hunyuan3DPaintPipeline.from_pretrained("$TEXTURE_MODEL_PATH")
print("Texture pipeline loaded successfully!")
EOF

echo "Rebuild/test complete. If the above message indicates success, your texture synthesis pipeline is ready."