import os
import datetime
import torch
from tqdm import tqdm
from PIL import PngImagePlugin
from diffusers import DiffusionPipeline

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

# Configuration via environment variables
save_path = os.getenv("OUTPUT_DIR", "lcm_images_1")
model_id = os.getenv("MODEL_ID", "SimianLuo/LCM_Dreamshaper_v7")
lcm_revision = os.getenv("LCM_REVISION", "fb9c5d")
lcm_custom_pipeline = os.getenv("LCM_CUSTOM_PIPELINE", "latent_consistency_txt2img")
lcm_custom_revision = os.getenv("LCM_CUSTOM_REVISION", "main")
default_steps = int(os.getenv("NUM_INFERENCE_STEPS", "8"))
default_guidance = float(os.getenv("GUIDANCE_SCALE", "30.0"))
default_lcm_origin_steps = int(os.getenv("LCM_ORIGIN_STEPS", "8"))
safety_checker = os.getenv("SAFETY_CHECKER", "disabled").lower()  # "disabled" or "default"

os.makedirs(save_path, exist_ok=True)

args = {} if safety_checker != "disabled" else {"safety_checker": None}

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline=lcm_custom_pipeline,
    custom_revision=lcm_custom_revision,
    revision=lcm_revision,
    **args,
)

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

    steps_input = input(f"Inference steps (default {default_steps}): ")
    num_inference_steps = int(steps_input) if steps_input.isdigit() else default_steps
    guidance_input = input(f"Guidance scale (default {default_guidance}): ")
    guidance_scale = float(guidance_input) if guidance_input.strip() else default_guidance
    lcm_origin_steps_input = input(f"LCM origin steps (default {default_lcm_origin_steps}): ")
    lcm_origin_steps = int(lcm_origin_steps_input) if lcm_origin_steps_input.isdigit() else default_lcm_origin_steps
    print(f"Generating {num_images} images for: '{prompt}'")

    for j in range(num_images):
        try:
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                lcm_origin_steps=lcm_origin_steps,
                output_type="pil",
            ).images
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
