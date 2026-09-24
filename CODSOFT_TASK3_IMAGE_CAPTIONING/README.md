# AI Image Caption Generator

## CodSoft Artificial Intelligence Internship - Task 3

An AI-powered image captioning application that analyzes an uploaded image and automatically generates a natural-language description.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- BLIP (Bootstrapping Language-Image Pre-training)
- Pillow
- Gradio

## How It Works

1. The user uploads an image.
2. The image is processed using the BLIP processor.
3. The pretrained BLIP model analyzes the image.
4. The model generates a text description.
5. The generated caption is displayed through the Gradio interface.

## Features

- Upload an image
- Automatic AI-generated caption
- Simple web interface
- Supports different types of images
- Uses a pretrained vision-language model

## Installation

Clone the repository and open the project folder.

Create and activate a virtual environment:

```bash
python -m venv venv