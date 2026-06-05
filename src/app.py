import os
import random
import tempfile

import numpy as np
import streamlit as st

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
QUANTIZATION_OPTIONS = [None, 4, 8]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each family groups one or more concrete variants. A variant fully describes
# how to build and drive the underlying mflux model class:
#
#   builder   - which mflux class to instantiate (see load_model)
#   alias     - ModelConfig name passed to ModelConfig.from_name (None for DepthPro)
#   task      - "generate" (txt2img/edit), "upscale" or "depth"
#   steps     - default number of inference steps
#   guidance  - whether a guidance slider is meaningful
#   gdef      - default guidance value
#   neg       - whether a negative prompt is supported
#   lora      - whether LoRA files can be applied
#   ref       - reference image: "none" | "optional" (img2img) | "required" (edit)
#   multi     - whether the variant accepts multiple reference images
MODEL_FAMILIES = {
    "FLUX.1": {
        "description": "Aug 2024 · 12B · Legacy FLUX.1. Decent quality with img2img, "
        "Kontext editing and ControlNet (Canny) support.",
        "variants": {
            "Schnell — fast txt2img": dict(builder="flux1", alias="schnell", task="generate", steps=4, guidance=False, gdef=0.0, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
            "Dev — txt2img": dict(builder="flux1", alias="dev", task="generate", steps=20, guidance=True, gdef=3.5, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
            "Krea-Dev — txt2img": dict(builder="flux1", alias="krea-dev", task="generate", steps=25, guidance=True, gdef=4.5, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
            "Dev — Kontext (edit)": dict(builder="flux1_kontext", alias="dev-kontext", task="generate", steps=20, guidance=True, gdef=2.5, neg=False, lora=True, ref="required", multi=False),  # noqa: E501
            "Dev — ControlNet Canny": dict(builder="flux1_cn", alias="dev-controlnet-canny", task="generate", steps=20, guidance=True, gdef=3.5, neg=False, lora=True, ref="required", multi=False),  # noqa: E501
            "Schnell — ControlNet Canny": dict(builder="flux1_cn", alias="schnell-controlnet-canny", task="generate", steps=4, guidance=False, gdef=0.0, neg=False, lora=True, ref="required", multi=False),  # noqa: E501
        },
    },
    "FLUX.2": {
        "description": "Jan 2026 · 4B & 9B · Fastest and smallest with very good quality "
        "and edit capabilities.",
        "variants": {
            "Klein 4B — txt2img": dict(builder="flux2", alias="flux2-klein-4b", task="generate", steps=4, guidance=True, gdef=4.0, neg=False, lora=True, ref="optional", multi=False),  # noqa: E501
            "Klein 9B — txt2img": dict(builder="flux2", alias="flux2-klein-9b", task="generate", steps=4, guidance=True, gdef=4.0, neg=False, lora=True, ref="optional", multi=False),  # noqa: E501
            "Klein Base 4B — txt2img": dict(builder="flux2", alias="flux2-klein-base-4b", task="generate", steps=28, guidance=True, gdef=4.0, neg=False, lora=True, ref="optional", multi=False),  # noqa: E501
            "Klein Base 9B — txt2img": dict(builder="flux2", alias="flux2-klein-base-9b", task="generate", steps=28, guidance=True, gdef=4.0, neg=False, lora=True, ref="optional", multi=False),  # noqa: E501
            "Klein 4B — edit": dict(builder="flux2_edit", alias="flux2-klein-4b", task="generate", steps=4, guidance=True, gdef=4.0, neg=False, lora=True, ref="required", multi=True),  # noqa: E501
            "Klein 9B — edit": dict(builder="flux2_edit", alias="flux2-klein-9b", task="generate", steps=4, guidance=True, gdef=4.0, neg=False, lora=True, ref="required", multi=True),  # noqa: E501
        },
    },
    "Z-Image": {
        "description": "Nov 2025 · 6B · Fast, small, very good quality and realism.",
        "variants": {
            "Z-Image Turbo": dict(builder="zimage", alias="z-image-turbo", task="generate", steps=4, guidance=False, gdef=0.0, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
            "Z-Image (base)": dict(builder="zimage", alias="z-image", task="generate", steps=20, guidance=True, gdef=3.5, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
        },
    },
    "FIBO": {
        "description": "Oct 2025 · 8B · Very good JSON-based prompt understanding. "
        "Accepts natural language or a JSON prompt. Has edit capabilities.",
        "variants": {
            "FIBO": dict(builder="fibo", alias="fibo", task="generate", steps=50, guidance=True, gdef=4.0, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
            "FIBO-lite (distilled)": dict(builder="fibo", alias="fibo-lite", task="generate", steps=50, guidance=False, gdef=1.0, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
            "FIBO-Edit": dict(builder="fibo_edit", alias="fibo-edit", task="generate", steps=50, guidance=True, gdef=4.0, neg=True, lora=True, ref="required", multi=False),  # noqa: E501
        },
    },
    "Qwen-Image": {
        "description": "Aug 2025 · 20B · Large (slower); strong prompt understanding and "
        "world knowledge. Has edit capabilities.",
        "variants": {
            "Qwen-Image — txt2img": dict(builder="qwen", alias="qwen-image", task="generate", steps=20, guidance=True, gdef=4.0, neg=True, lora=True, ref="optional", multi=False),  # noqa: E501
            "Qwen-Image-Edit": dict(builder="qwen_edit", alias="qwen-image-edit", task="generate", steps=20, guidance=True, gdef=4.0, neg=True, lora=True, ref="required", multi=True),  # noqa: E501
        },
    },
    "SeedVR2": {
        "description": "Jun 2025 · 3B & 7B · Best upscaling model. Upload an image and "
        "choose a target resolution or scale factor.",
        "variants": {
            "SeedVR2 3B": dict(builder="seedvr2", alias="seedvr2-3b", task="upscale", steps=1, guidance=False, gdef=0.0, neg=False, lora=False, ref="required", multi=False),  # noqa: E501
            "SeedVR2 7B": dict(builder="seedvr2", alias="seedvr2-7b", task="upscale", steps=1, guidance=False, gdef=0.0, neg=False, lora=False, ref="required", multi=False),  # noqa: E501
        },
    },
    "Depth Pro": {
        "description": "Oct 2024 · Apple's very fast and accurate depth estimation model. "
        "Upload an image to produce a depth map.",
        "variants": {
            "Depth Pro": dict(builder="depth", alias=None, task="depth", steps=0, guidance=False, gdef=0.0, neg=False, lora=False, ref="required", multi=False),  # noqa: E501
        },
    },
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = None
if "generating" not in st.session_state:
    st.session_state.generating = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_uploaded(uploaded_file) -> str:
    """Persist an uploaded file to a temp path and return that path."""
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


@st.cache_resource(show_spinner=False)
def load_model(builder, alias, quantize, lora_paths, lora_scales):
    """Instantiate the appropriate mflux model class for the chosen variant.

    Imports are deferred so that only the selected model family's code is
    imported, keeping startup light. Cached so repeated generations with the
    same configuration reuse the loaded weights.
    """
    from mflux.models.common.config.model_config import ModelConfig

    model_config = ModelConfig.from_name(alias) if alias else None
    lora_paths = list(lora_paths) or None
    lora_scales = list(lora_scales) or None

    if builder == "flux1":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        return Flux1(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    if builder == "flux1_kontext":
        from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

        return Flux1Kontext(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    if builder == "flux1_cn":
        from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet

        return Flux1Controlnet(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)  # noqa: E501

    if builder == "flux2":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        return Flux2Klein(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    if builder == "flux2_edit":
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

        return Flux2KleinEdit(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)  # noqa: E501

    if builder == "zimage":
        from mflux.models.z_image.variants.z_image import ZImage

        return ZImage(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    if builder == "fibo":
        from mflux.models.fibo.variants.txt2img.fibo import FIBO

        return FIBO(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    if builder == "fibo_edit":
        from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit

        return FIBOEdit(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    if builder == "qwen":
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    if builder == "qwen_edit":
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        return QwenImageEdit(model_config=model_config, quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)  # noqa: E501

    if builder == "seedvr2":
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

        return SeedVR2(model_config=model_config, quantize=quantize)

    if builder == "depth":
        from mflux.models.depth_pro.model.depth_pro import DepthPro

        return DepthPro(quantize=quantize)

    raise ValueError(f"Unknown builder: {builder}")


def run_generation(model, builder, params):
    """Drive the loaded model and return a PIL image."""
    p = params

    if builder == "flux1":
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], num_inference_steps=p["steps"],
            height=p["height"], width=p["width"], guidance=p["guidance"],
            image_path=p["ref_paths"][0] if p["ref_paths"] else None,
            image_strength=p["image_strength"] if p["ref_paths"] else None,
            negative_prompt=p["negative_prompt"],
        )
        return result.image

    if builder == "flux1_kontext":
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], num_inference_steps=p["steps"],
            height=p["height"], width=p["width"], guidance=p["guidance"],
            image_path=p["ref_paths"][0], image_strength=p["image_strength"],
        )
        return result.image

    if builder == "flux1_cn":
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], controlnet_image_path=p["ref_paths"][0],
            num_inference_steps=p["steps"], height=p["height"], width=p["width"],
            guidance=p["guidance"], controlnet_strength=p["controlnet_strength"],
        )
        return result.image

    if builder == "flux2":
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], num_inference_steps=p["steps"],
            height=p["height"], width=p["width"], guidance=p["guidance"],
            image_path=p["ref_paths"][0] if p["ref_paths"] else None,
            image_strength=p["image_strength"] if p["ref_paths"] else None,
        )
        return result.image

    if builder == "flux2_edit":
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], num_inference_steps=p["steps"],
            height=p["height"], width=p["width"], guidance=p["guidance"],
            image_paths=p["ref_paths"], image_strength=p["image_strength"],
        )
        return result.image

    if builder in ("zimage", "fibo", "qwen"):
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], num_inference_steps=p["steps"],
            height=p["height"], width=p["width"], guidance=p["guidance"],
            image_path=p["ref_paths"][0] if p["ref_paths"] else None,
            image_strength=p["image_strength"] if p["ref_paths"] else None,
            negative_prompt=p["negative_prompt"],
        )
        return result.image

    if builder == "fibo_edit":
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], image_path=p["ref_paths"][0],
            num_inference_steps=p["steps"], height=p["height"], width=p["width"],
            guidance=p["guidance"], negative_prompt=p["negative_prompt"],
        )
        return result.image

    if builder == "qwen_edit":
        result = model.generate_image(
            seed=p["seed"], prompt=p["prompt"], image_paths=p["ref_paths"],
            num_inference_steps=p["steps"], guidance=p["guidance"],
            negative_prompt=p["negative_prompt"],
        )
        return result.image

    if builder == "seedvr2":
        from mflux.utils.scale_factor import ScaleFactor

        res_text = str(p["resolution"]).strip()
        resolution = ScaleFactor.parse(res_text) if res_text.lower().endswith("x") else int(res_text)
        result = model.generate_image(
            seed=p["seed"], image_path=p["ref_paths"][0],
            resolution=resolution, softness=p["softness"],
        )
        return result.image

    if builder == "depth":
        return model.create_depth_map(image_path=p["ref_paths"][0]).depth_image

    raise ValueError(f"Unknown builder: {builder}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MFLUX Studio", layout="centered")
    st.title("MFLUX Studio")
    st.markdown("Image generation, editing, upscaling and depth estimation on Apple Silicon via MFLUX.")

    disabled = st.session_state.generating

    examples = [
        "a tiny astronaut hatching from an egg on the moon",
        "a cat holding a sign that says hello world",
        "an anime illustration of a wiener schnitzel",
    ]

    with st.sidebar:
        with st.expander("Model Configuration", expanded=True):
            family_name = st.selectbox("Model family", list(MODEL_FAMILIES.keys()), disabled=disabled)
            family = MODEL_FAMILIES[family_name]
            st.caption(family["description"])

            variant_name = st.selectbox("Variant", list(family["variants"].keys()), disabled=disabled)
            variant = family["variants"][variant_name]

            selected_quantization = st.selectbox("Quantization", QUANTIZATION_OPTIONS, disabled=disabled)

        # LoRA upload (only for families that support it)
        lora_paths, lora_scales = [], []
        if variant["lora"]:
            with st.expander("LoRA files"):
                lora_uploaders = st.file_uploader(
                    "Upload LoRA .safetensors files",
                    type=["safetensors"],
                    accept_multiple_files=True,
                    disabled=disabled,
                    key="uploader",
                )
                for idx, lora_file in enumerate(lora_uploaders or []):
                    scale = st.slider(
                        f"LoRA {idx + 1}: {lora_file.name}", 0.0, 1.0, 1.0,
                        disabled=disabled, key=f"slider_{idx}",
                    )
                    lora_scales.append(scale)
                    lora_paths.append(_save_uploaded(lora_file))

        # Reference / source image upload
        ref_paths = []
        if variant["ref"] != "none":
            label = {
                "depth": "Source image for depth estimation",
                "upscale": "Image to upscale",
            }.get(variant["task"], "Reference image" + (" (required)" if variant["ref"] == "required" else " (optional, img2img)"))  # noqa: E501
            with st.expander("Input image", expanded=variant["ref"] == "required"):
                if variant["multi"]:
                    uploads = st.file_uploader(
                        label, type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True, disabled=disabled, key="ref_multi",
                    )
                    for up in uploads or []:
                        ref_paths.append(_save_uploaded(up))
                else:
                    up = st.file_uploader(
                        label, type=["png", "jpg", "jpeg"], disabled=disabled, key="ref_single",
                    )
                    if up is not None:
                        ref_paths.append(_save_uploaded(up))
                for path in ref_paths:
                    st.image(path, use_container_width=True)

        # Generation parameters
        randomize_seed = True
        seed = 0
        controlnet_strength = 1.0
        image_strength = 0.4
        resolution = 384
        softness = 0.0
        width = height = 1024
        steps = variant["steps"]
        guidance = variant["gdef"]

        with st.expander("Advanced Settings", expanded=False):
            randomize_seed = st.checkbox("Randomize seed", value=True, disabled=disabled)
            seed = (
                random.randint(0, MAX_SEED)
                if randomize_seed
                else st.slider("Seed", 0, MAX_SEED, 0, disabled=disabled)
            )

            if variant["task"] == "upscale":
                res_text = st.text_input(
                    "Target resolution (shortest edge px) or scale factor like '2x'",
                    value="2x", disabled=disabled,
                )
                resolution = res_text.strip()
                softness = st.slider("Softness", 0.0, 1.0, 0.0, disabled=disabled)
            elif variant["task"] == "generate":
                width = st.slider("Width", 256, MAX_IMAGE_SIZE, 1024, 32, disabled=disabled)
                height = st.slider("Height", 256, MAX_IMAGE_SIZE, 1024, 32, disabled=disabled)
                steps = st.slider("Inference steps", 1, 50, variant["steps"], disabled=disabled)
                if variant["guidance"]:
                    guidance = st.slider("Guidance", 0.0, 10.0, variant["gdef"], disabled=disabled)
                if variant["ref"] == "controlnet" or variant["builder"] == "flux1_cn":
                    controlnet_strength = st.slider("ControlNet strength", 0.0, 1.0, 0.4, disabled=disabled)
                elif variant["ref"] != "none" and variant["builder"] not in ("flux1_cn",):
                    image_strength = st.slider(
                        "Image strength (img2img / edit influence)", 0.0, 1.0, 0.4, disabled=disabled,
                    )

    # Main interface
    needs_prompt = variant["task"] == "generate"
    prompt = ""
    negative_prompt = None

    if needs_prompt:
        if not st.session_state.prompt_input or st.session_state.generating:
            st.markdown("### Try these examples:")
            cols = st.columns(len(examples))
            for col, example in zip(cols, examples):
                if col.button(example, use_container_width=True, disabled=disabled):
                    st.session_state.prompt_input = example

        col1, col2 = st.columns([3, 1])
        with col1:
            prompt = st.text_area("Enter your prompt", key="prompt_input", disabled=disabled)
        with col2:
            st.write("")
            st.write("")
            st.write("")
            generate_button = st.button("Generate", use_container_width=True, disabled=disabled)
        if variant["neg"]:
            negative_prompt = st.text_input("Negative prompt (optional)", disabled=disabled) or None
    else:
        action = "Upscale" if variant["task"] == "upscale" else "Estimate depth"
        generate_button = st.button(action, use_container_width=True, disabled=disabled)

    # Validate and run
    if generate_button:
        if needs_prompt and not prompt:
            st.warning("Please enter a prompt.")
            return
        if variant["ref"] == "required" and not ref_paths:
            st.warning("This model requires an input image. Please upload one in the sidebar.")
            return

        params = dict(
            seed=seed, prompt=prompt, negative_prompt=negative_prompt,
            steps=steps, width=width, height=height, guidance=guidance,
            ref_paths=ref_paths, image_strength=image_strength,
            controlnet_strength=controlnet_strength,
            resolution=resolution, softness=softness,
        )

        st.session_state.generating = True
        try:
            with st.spinner(f"Loading {variant_name}…"):
                model = load_model(
                    variant["builder"], variant["alias"], selected_quantization,
                    tuple(lora_paths), tuple(lora_scales),
                )
            spinner_msg = {
                "upscale": "Upscaling image…",
                "depth": "Estimating depth…",
            }.get(variant["task"], "Generating image…")
            with st.spinner(spinner_msg):
                image = run_generation(model, variant["builder"], params)
            if image is not None:
                caption = {
                    "upscale": "Upscaled image",
                    "depth": "Depth map",
                }.get(variant["task"], "Generated image")
                st.image(image, caption=caption, use_container_width=True)
        except Exception as exc:  # surface errors in the UI instead of only the console
            st.error(f"Generation failed: {exc}")
        finally:
            st.session_state.generating = False
    elif needs_prompt and not prompt:
        st.info("Enter a prompt above or click an example to generate an image.")


if __name__ == "__main__":
    main()
