from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import gradio as gr


# -------------------------------------------------
# Load AI Model
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.to(device)


# -------------------------------------------------
# Generate Caption
# -------------------------------------------------

def generate_caption(image):

    if image is None:
        return "Please upload an image first."

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    output = model.generate(
        **inputs,
        max_new_tokens=50
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


# -------------------------------------------------
# Custom Interface
# -------------------------------------------------

custom_css = """
body {
    background: linear-gradient(135deg, #07111f, #101a33);
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.title {
    text-align: center;
    font-size: 48px !important;
    font-weight: 800 !important;
    margin-top: 20px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px !important;
    margin-bottom: 30px;
}

.caption-box textarea {
    font-size: 30px !important;
    line-height: 1.5 !important;
    font-weight: 600 !important;
    padding: 25px !important;
}

.generate-btn {
    font-size: 20px !important;
    font-weight: bold !important;
    min-height: 60px !important;
}

.info-box {
    text-align: center;
    font-size: 16px;
    padding: 15px;
    margin-top: 20px;
}

.footer {
    text-align: center;
    font-size: 15px;
    margin-top: 25px;
    opacity: 0.8;
}
"""


# -------------------------------------------------
# Gradio Application
# -------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.HTML(
        """
        <div class="title">
            🤖 AI Image Caption Generator
        </div>

        <div class="subtitle">
            Upload an image and let AI generate a meaningful description
        </div>
        """
    )

    with gr.Row():

        # Left side - Image
        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="📷 Upload Image"
            )

            generate_button = gr.Button(
                "✨ Generate Caption",
                variant="primary",
                elem_classes="generate-btn"
            )

        # Right side - Caption
        with gr.Column():

            caption_output = gr.Textbox(
                label="✨ Generated Caption",
                placeholder="Your AI-generated caption will appear here...",
                lines=6,
                elem_classes="caption-box"
            )

            gr.Markdown(
                """
                ### 💡 About the Model

                This application uses **Salesforce BLIP**, a
                pretrained vision-language model for image captioning.
                """
            )

    generate_button.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=caption_output
    )

    gr.HTML(
        """
        <div class="info-box">
            🔵 Powered by Salesforce BLIP
            (Bootstrapping Language-Image Pre-training)
        </div>

        <div class="footer">
            Made with ❤️ using Python, Gradio & Hugging Face Transformers
        </div>
        """
    )


# -------------------------------------------------
# Start Application
# -------------------------------------------------

if __name__ == "__main__":
    demo.launch()