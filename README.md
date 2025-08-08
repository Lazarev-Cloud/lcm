# Latent Consistency Model (LCM) — simple web and CLI

Based on https://github.com/0xbitches/sd-webui-lcm

Really simple implementation for batch image generation, now with a minimal web UI and Docker-first workflow.

# Text-to-Image Generation

This script is designed to generate images from text prompts using a text-to-image diffusion model. It allows users to input prompts and specify the number of images to generate, and then saves these images with relevant metadata.

## Features

- **Text-to-Image Conversion**: Transforms text prompts into vivid images using a deep learning model.
- **Custom Image Generation**: Users can specify the number of images they want to generate for each prompt.
- **Automatic CPU/GPU Selection**: Automatically detects and utilizes GPU (CUDA) if available, for enhanced performance. Defaults to CPU when GPU is not available.
- **GPU Acceleration**: Utilizes CUDA for GPU acceleration (if available), ensuring swift image generation.
- **Descriptive Filenames**: Generates informative file names based on the prompt, timestamp, and a snippet of the prompt, making them easily identifiable.
- **Metadata Storage and Embedding**: Each image is saved with important metadata, including the prompt used, and the number of inference steps.
- **Progress Tracking and Updates**: A progress bar displays real-time updates as images are being saved, keeping users informed about the process.
- **Robust Error Handling**: Ensures a smoother user experience by gracefully handling errors during image generation.


# Image Generator Script

This script leverages a state-of-the-art text-to-image diffusion model to generate stunning visual representations from textual prompts. It's designed to be intuitive and user-friendly, making the process of creating images from text both efficient and enjoyable.



## Quick start (Docker)

CPU-only (works everywhere):

```bash
docker build -t lcm:cpu .
docker run --rm -p 8000:8000 -v %cd%/outputs:/data lcm:cpu
# Open http://localhost:8000
```

NVIDIA GPU (Windows/Linux with NVIDIA Container Toolkit):

```bash
docker build -t lcm:gpu --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .
docker run --rm -p 8000:8000 --gpus all -v %cd%/outputs:/data lcm:gpu
# Open http://localhost:8000
```

Environment overrides:

```bash
docker run --rm -p 8000:8000 -v %cd%/outputs:/data \
  -e MODEL_ID=SimianLuo/LCM_Dreamshaper_v7 \
  -e NUM_INFERENCE_STEPS=8 -e GUIDANCE_SCALE=30.0 -e LCM_ORIGIN_STEPS=8 \
  -e SAFETY_CHECKER=default \
  lcm:cpu
```

CLI mode (inside container):

```bash
docker run --rm -it -v %cd%/outputs:/data lcm:cpu python text2img.py
```

---

**Note**: This script is designed to be intuitive and easy to use. It leverates advanced machine learning techniques to bring your textual ideas to life in the form of images.

Enjoy creating beautiful images with just a few keystrokes!

---

## Local installation (optional)

If you prefer not to use Docker:

```bash
python -m venv .venv && .venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Requirements

The script requires the following libraries:
- `torch`: For handling deep learning operations.
- `tqdm`: To display progress bars.
- `Pillow`: For image processing.
- `diffusers`: For using the diffusion model.

Ensure your machine has CUDA support for GPU acceleration.

## Web usage

- Start the container (see Quick start)
- Open `http://localhost:8000`
- Enter your prompt, optionally tweak steps/guidance
- Images are saved into the mapped host `outputs/` directory

## User Interaction

- Enter text prompts as instructed.
- Specify the desired number of images for each prompt.
- Type 'q' at the prompt stage to exit the program.

## Output

- Images are saved in the `outputs/` directory when running via Docker (mapped to container `/data`). In CLI mode without Docker, defaults to `lcm_images_1`.
- Filenames contain the date, time, and a snippet of the prompt.
- Metadata includes the original prompt and the number of inference steps.

---

## Additional Note on Image Safety

In the script, the line:

```python
pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main", revision="fb9c5d", safety_checker=None)
```


initializes the diffusion model without a safety filter (`safety_checker=None`). This setting allows for the generation of images without any content restriction. However, if you prefer to generate only SFW (Safe For Work) content, you can enable the safety filter by removing the `safety_checker=None` parameter. This will apply the model's default safety checker, filtering out potentially NSFW (Not Safe For Work) content.

To ensure SFW content, modify the line as follows:

```python
pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main", revision="fb9c5d")
```

By enabling the safety checker, you add an extra layer of content moderation to the image generation process.

## Docker notes and hardening

- Uses `python:3.10-slim` minimal base
- Non-root user `appuser`
- Healthcheck at `/healthz`
- Caching optimized by installing dependencies before copying source
- `.dockerignore` recommended for smaller builds

## GitHub Actions and Releases

This repo ships with GitHub Actions to:

- Build and push Docker images to GHCR on pushes to `main` (`.github/workflows/docker-publish.yml`).
  - CPU image tags follow branch/tag; `latest` on default branch, plus a `gpu` tag for CUDA wheels.
- Create GitHub Releases when pushing tags like `v1.0.0` (`.github/workflows/create-release.yml`).
  - Uses GitHub’s automatic release notes.
- Update release notes content on publish (`.github/workflows/release-notes.yml`).

After a `main` push, pull with:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
docker pull ghcr.io/<owner>/<repo>:gpu
```

Make sure your repo visibility allows GHCR pulls, or authenticate: `docker login ghcr.io`.

---
