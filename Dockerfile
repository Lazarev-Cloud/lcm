# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install git, required for cloning repositories or fetching specific revisions
RUN apt-get update && apt-get install -y git

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the model weights
RUN python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('SimianLuo/LCM_Dreamshaper_v7', custom_pipeline='latent_consistency_txt2img', custom_revision='main', revision='fb9c5d', safety_checker=None)"

# Run app.py when the container launches
CMD ["python", "./app.py"]
