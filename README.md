# Latent Consistency Model in terminal

Based on https://github.com/0xbitches/sd-webui-lcm

Really simple implementation for batch image generation

# Text-to-Image Generation Script

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



## How to Use

1. **Start the Script**: Run the script in your Python environment.
2. **Enter Prompts**: When prompted, enter the text you wish to transform into an image.
3. **Specify Image Quantity**: After entering a prompt, you'll be asked to specify the number of images you want to generate. If you don't specify a number, it will default to 1.
4. **View Generated Images**: Images will be saved in a specified directory, complete with relevant metadata.

---

**Note**: This script is designed to be intuitive and easy to use. It leverates advanced machine learning techniques to bring your textual ideas to life in the form of images.

Enjoy creating beautiful images with just a few keystrokes!

---

## Installation

To use this script, you need to have Python installed on your system, along with several libraries. Here are the steps to set it up:

1. **Install Python**: Ensure that you have Python 3.6 or later installed. You can download it from [python.org](https://www.python.org/downloads/).

2. **Clone the Repository**: Clone or download the repository containing the script to your local machine. (or just [download](https://github.com/Lazarev-Cloud/lcm/archive/refs/heads/main.zip) it and unzip)

3. **Install Dependencies**: Open your terminal or command prompt and navigate to the script's directory. Install the required libraries using pip:

    ```bash
    pip install torch tqdm Pillow diffusers transformers
    ```

## Requirements

The script requires the following libraries:
- `torch`: For handling deep learning operations.
- `tqdm`: To display progress bars.
- `Pillow`: For image processing.
- `diffusers`: For using the diffusion model.

Ensure your machine has CUDA support for GPU acceleration.

## Usage

1. **Run the Script**: Execute the script in your Python environment.
2. **Enter Prompts**: When prompted, enter the text you wish to turn into an image.
3. **Specify Image Quantity**: Indicate how many images you want for each prompt. Defaults to 1 if unspecified.
4. **View Generated Images**: Images will be saved in the `lcm_images_1` directory, complete with metadata.

## User Interaction

- Enter text prompts as instructed.
- Specify the desired number of images for each prompt.
- Type 'q' at the prompt stage to exit the program.

## Output

- Images are saved in the `lcm_images_1` directory.
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

---
