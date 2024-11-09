import streamlit as st
import numpy as np
import random
from mflux import Flux1, Flux1Controlnet, ModelConfig, Config, ConfigControlnet

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = st.session_state.get('prompt_input', None)
if 'generating' not in st.session_state:
    st.session_state.generating = False


def main():
    st.set_page_config(page_title="FLUX.1", layout="centered")

    # Title and description
    st.title("FLUX.1")
    st.markdown("""
    12B parameter rectified flow transformer distilled from FLUX.1 pro for image generation.
    """)

    # Example prompts
    examples = [
        "a tiny astronaut hatching from an egg on the moon",
        "a cat holding a sign that says hello world",
        "an anime illustration of a wiener schnitzel",
    ]

    # Sidebar for advanced settings
    with st.sidebar:
        with st.expander("Model Configuration", expanded=True):
            model_options = ["Schnell", "Dev"]
            selected_model = st.selectbox("Model", model_options, disabled=st.session_state.generating)

            quantization_options = [None, 4, 8]
            selected_quantization = st.selectbox("Quantization", quantization_options,
                                                 disabled=st.session_state.generating)

        with st.expander("Upload and Configure LoRA Files", expanded=True):
            lora_uploaders = st.file_uploader(
                "Upload LoRA .safetensor files",
                type=["safetensors"],
                accept_multiple_files=True,
                disabled=st.session_state.generating,
                key="uploader",
                label_visibility="hidden"
            )

            lora_scales = []
            if lora_uploaders:
                for idx, lora_file in enumerate(lora_uploaders):
                    lora_scale = st.slider(f"LoRA File {idx+1}: {lora_file.name}", 0.0,
                                           1.0, 1.0, disabled=st.session_state.generating, key=f"slider_{idx}")
                    lora_scales.append(lora_scale)

            lora_paths = []
            for lora_file in lora_uploaders:
                if lora_file:
                    import tempfile
                    import os
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, lora_file.name)
                    lora_paths.append(temp_file_path)
                    with open(temp_file_path, "wb") as f:
                        f.write(lora_file.getbuffer())

        with st.expander("Advanced Settings"):
            randomize_seed = st.checkbox("Randomize seed", value=True, disabled=st.session_state.generating)
            seed = st.slider("Seed", 0, MAX_SEED, 0,
                             disabled=st.session_state.generating) if not randomize_seed else random.randint(0, MAX_SEED)

            width = st.slider("Width", 256, MAX_IMAGE_SIZE, 1024, 32, disabled=st.session_state.generating)
            height = st.slider("Height", 256, MAX_IMAGE_SIZE, 1024, 32, disabled=st.session_state.generating)
            num_inference_steps = st.slider("Number of inference steps", 1, 50, 4, disabled=st.session_state.generating)
            guidance = st.slider("Guidance", 0.0, 7.0, 3.5, disabled=st.session_state.generating)

        controlnet_image_path = None
        with st.expander("ControlNet Settings"):
            uploaded_image = st.file_uploader("Upload ControlNet image", type=[
                                              "png", "jpg", "jpeg"], disabled=st.session_state.generating)
            if uploaded_image is not None:
                import tempfile
                import os

                temp_dir = tempfile.gettempdir()
                controlnet_image_path = os.path.join(temp_dir, uploaded_image.name)

                with open(controlnet_image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

                st.image(controlnet_image_path, caption="Uploaded ControlNet image", use_container_width=True)
                controlnet_strength = st.slider("Strength", 0.0, 1.0, 0.4, disabled=st.session_state.generating)

    # Example prompts section
    if not st.session_state.prompt_input or st.session_state.generating:
        st.markdown("### Try these examples:")
        example_cols = st.columns(len(examples))
        for col, example in zip(example_cols, examples):
            if col.button(example, use_container_width=True, disabled=st.session_state.generating):
                st.session_state.prompt_input = example
                prompt = example
    # Main interface
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area("Enter your prompt", key="prompt_input", disabled=st.session_state.generating)
    with col2:
        st.write("")  # Add an empty line to push the button to the middle
        st.write("")  # Add another empty line to push the button to the middle
        st.write("")
        generate_button = st.button("Generate", use_container_width=True, disabled=st.session_state.generating)
    # Generate image when button is clicked
    if generate_button and prompt:
        st.session_state.generating = True
        pipe = load_model(selected_model.lower(), selected_quantization, lora_paths, lora_scales, controlnet_image_path)
        with st.spinner("Generating image..." if controlnet_image_path is None else "Generating image using ControlNet..."):
            generated_image = None
            if(controlnet_image_path is not None):
                print(controlnet_image_path, controlnet_strength)
                generated_image = pipe.generate_image(
                    seed=seed,
                    prompt=prompt,
                    output='',
                    controlnet_image_path=controlnet_image_path,
                    config=ConfigControlnet(
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                        guidance=guidance,
                        controlnet_strength=controlnet_strength
                    ),
                )
            else:
                generated_image = pipe.generate_image(
                    seed=seed,
                    prompt=prompt,
                    config=Config(
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                        guidance=guidance
                    ),
                )
            if generated_image:
                st.image(generated_image.image, caption="Generated Image", use_container_width=True)

        st.session_state.generating = False
    # Instructions if no prompt
    elif not prompt:
        st.info("Enter a prompt above or click one of the example prompts to generate an image.")

# Model setup


@st.cache_resource
def load_model(selected_model, quantization=None, lora_paths=[], lora_scales=[], controlnet_image_path=None) -> Flux1 | Flux1Controlnet:
    if(controlnet_image_path != None):
        return Flux1Controlnet(
            model_config=ModelConfig.from_alias(selected_model),
            quantize=quantization,
            local_path=None,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    return Flux1(model_config=ModelConfig.from_alias(selected_model),
                 quantize=quantization,
                 lora_paths=lora_paths,
                 lora_scales=lora_scales
                 )


if __name__ == "__main__":
    main()
