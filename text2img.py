import os
import torch
from tqdm import tqdm
from PIL import PngImagePlugin
from diffusers import DiffusionPipeline
import datetime

# Function to save a single image with metadata
def save_image(image, filename, metadata: dict):
    meta_tuples = [(k, str(v)) for k, v in metadata.items()]
    png_info = PngImagePlugin.PngInfo()
    for k, v in meta_tuples:
        png_info.add_text(k, v)
    image.save(filename, pnginfo=png_info)

# Improved File Naming
def generate_filename(prompt, timestamp, index):
    prompt_snippet = "_".join(prompt.split()[:3])  # Takes first 3 words of the prompt
    return f"{timestamp}_{prompt_snippet}_{index}.png"

# Save Path
save_path = "lcm_images_1"
os.makedirs(save_path, exist_ok=True)

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main", revision="fb9c5d", safety_checker=None)

# Check if CUDA (GPU support) is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else None
print(f"Using device: {device}")

# Set the device and dtype for the pipeline
pipe.to(torch_device=device, torch_dtype=torch_dtype)

# Main Loop
while True:
    prompt = input("Enter prompt (or 'q' to quit): ")
    if prompt.lower() == "q":
        break

    count = input("Number of images (default 1): ")
    num_images = int(count) if count.isdigit() else 1

    num_inference_steps = 8  # Can be adjusted or made user-input
    print(f"Generating {num_images} images for: '{prompt}'")

    for j in range(num_images):
        try:
            images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=30.0, lcm_origin_steps=8, output_type="pil").images
        except Exception as e:
            print(f"Error generating image: {e}")
            continue

        metadata = {"prompt": prompt, "num_steps": num_inference_steps}

        for i, image in enumerate(tqdm(images, desc="Saving images")):
            timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
            filename = generate_filename(prompt, timestamp, i)
            output_path = os.path.join(save_path, filename)
            save_image(image, output_path, metadata)

    print(f"Images saved to {save_path}")

print("Image generation completed.")
