#!/usr/bin/env python3
"""
ComfyUI image generation backend.

Configuration keys:
  COMFYUI_BASE_URL       (optional, default http://127.0.0.1:8188)
  COMFYUI_API_KEY        (optional, for hosted/authenticated ComfyUI instances)
  COMFYUI_UNET           (optional, default "z_image_turbo_nvfp4.safetensors")
  COMFYUI_CLIP           (optional, default "qwen_3_4b.safetensors")
  COMFYUI_VAE            (optional, default "ae.safetensors")
  COMFYUI_SYSTEM_PROMPT  (optional, default "superior")
  COMFYUI_NEGATIVE_PROMPT(optional, default: generic text/watermark exclusion prompt)
  COMFYUI_WORKFLOW       (optional, path to a custom ComfyUI API-format workflow JSON.
                           If set, the script looks for CLIPTextEncode/CLIPTextEncodeLumina2
                           nodes titled "positive"/"negative" (via _meta.title) to inject the
                           prompt, plus EmptySD3LatentImage/EmptyLatentImage for size and
                           KSampler for seed. If not found, falls back to the built-in
                           default workflow.)

The default workflow targets a Lumina2-style pipeline (Z-Image-Turbo UNET + Qwen CLIP,
AuraFlow model sampling, SD3-style empty latent, res_multistep/simple sampling), matching
the reference ComfyUI graph this backend was ported from.
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from console_encoding import configure_utf8_stdio  # noqa: E402

configure_utf8_stdio()

if __name__ == "__main__":
    print(__doc__)
    print("Use via: python3 skills/ppt-master/scripts/image_gen.py \"prompt\" --backend comfyui")
    raise SystemExit(0 if any(arg in {"-h", "--help", "help"} for arg in sys.argv[1:]) else 1)

import json
import os
import random
import time
import uuid

import requests

from image_backends.backend_common import (
    MAX_RETRIES,
    download_image,
    http_error,
    is_rate_limit_error,
    normalize_image_size,
    resolve_output_path,
    retry_delay,
)


DEFAULT_BASE_URL = "http://comfyui:8188"
DEFAULT_UNET = "z_image_turbo_nvfp4.safetensors"
DEFAULT_CLIP = "qwen_3_4b.safetensors"
DEFAULT_CLIP_TYPE = "lumina2"
DEFAULT_VAE = "ae.safetensors"
DEFAULT_SYSTEM_PROMPT = "superior"
DEFAULT_NEGATIVE_PROMPT = (
    "text, letters, words, writing, signature, watermark, logo, title, 'Prompt Start'"
)
DEFAULT_MODEL_SHIFT = 3
DEFAULT_STEPS = 12
DEFAULT_CFG = 2
DEFAULT_SAMPLER_NAME = "res_multistep"
DEFAULT_SCHEDULER = "simple"
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 300

# Target pixel dimensions (width*height) per logical size preset.
# EmptySD3LatentImage wants multiples of 16; presets below stay within that.
ASPECT_RATIO_SIZE_MAP = {
    "512px": {
        "1:1": (512, 512),
        "2:3": (448, 640),
        "3:2": (640, 448),
        "3:4": (448, 576),
        "4:3": (576, 448),
        "4:5": (448, 576),
        "5:4": (576, 448),
        "9:16": (384, 672),
        "16:9": (672, 384),
        "21:9": (768, 320),
    },
    "1K": {
        "1:1": (1024, 1024),
        "2:3": (832, 1216),
        "3:2": (1216, 832),
        "3:4": (896, 1152),
        "4:3": (1152, 896),
        "4:5": (896, 1088),
        "5:4": (1088, 896),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
        "21:9": (1536, 640),
    },
    "2K": {
        "1:1": (1536, 1536),
        "2:3": (1248, 1824),
        "3:2": (1824, 1248),
        "3:4": (1344, 1728),
        "4:3": (1728, 1344),
        "4:5": (1344, 1632),
        "5:4": (1632, 1344),
        "9:16": (1152, 2016),
        "16:9": (2016, 1152),
        "21:9": (2304, 960),
    },
    "4K": {
        "1:1": (2048, 2048),
        "2:3": (1664, 2432),
        "3:2": (2432, 1664),
        "3:4": (1792, 2304),
        "4:3": (2304, 1792),
        "4:5": (1792, 2176),
        "5:4": (2176, 1792),
        "9:16": (1536, 2688),
        "16:9": (2688, 1536),
        "21:9": (3072, 1280),
    },
}


def _resolve_url(base_url: str, path: str) -> str:
    """Join a ComfyUI base URL with an API path."""
    return base_url.rstrip("/") + path


def _resolve_size(aspect_ratio: str, image_size: str) -> tuple:
    """Resolve (width, height) for a ratio and logical size preset."""
    normalized = normalize_image_size(image_size)
    size = (ASPECT_RATIO_SIZE_MAP.get(normalized) or {}).get(aspect_ratio)
    if not size:
        supported = sorted(ASPECT_RATIO_SIZE_MAP["1K"])
        raise ValueError(
            f"Unsupported aspect ratio '{aspect_ratio}' for ComfyUI backend. "
            f"Supported: {supported}"
        )
    # EmptySD3LatentImage wants multiples of 16.
    w, h = size
    return (w // 16) * 16, (h // 16) * 16


def _default_workflow(prompt: str, negative_prompt: str, width: int, height: int,
                       unet_name: str, clip_name: str, vae_name: str,
                       system_prompt: str, shift: float, steps: int, cfg: float,
                       sampler_name: str, scheduler: str, seed: int) -> dict:
    """Build the Lumina2 / Z-Image-Turbo ComfyUI API-format txt2img workflow graph."""
    return {
        "1": {
            "inputs": {"anything": ["6", 0]},
            "class_type": "easy cleanGpuUsed",
            "_meta": {"title": "Clean VRAM Used"},
        },
        "2": {
            "inputs": {"filename_prefix": "comfyui-image-gen", "images": ["1", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "3": {
            "inputs": {"clip_name": clip_name, "type": DEFAULT_CLIP_TYPE, "device": "default"},
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        "4": {
            "inputs": {"vae_name": vae_name},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "5": {
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        "6": {
            "inputs": {"samples": ["9", 0], "vae": ["4", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "7": {
            "inputs": {"system_prompt": system_prompt, "user_prompt": prompt, "clip": ["3", 0]},
            "class_type": "CLIPTextEncodeLumina2",
            "_meta": {"title": "positive"},
        },
        "9": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1,
                "model": ["10", 0],
                "positive": ["7", 0],
                "negative": ["12", 0],
                "latent_image": ["11", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "10": {
            "inputs": {"shift": shift, "model": ["5", 0]},
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "ModelSamplingAuraFlow"},
        },
        "11": {
            "inputs": {"width": width, "height": height, "batch_size": 1},
            "class_type": "EmptySD3LatentImage",
            "_meta": {"title": "Empty Latent Image (SD3)"},
        },
        "12": {
            "inputs": {"text": negative_prompt, "clip": ["3", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "negative"},
        },
    }


def _load_custom_workflow(workflow_path: str, prompt: str, negative_prompt: str,
                           width: int, height: int, seed: int) -> dict:
    """Load a user-provided ComfyUI API-format workflow and inject prompt/size/seed."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    for node in workflow.values():
        title = ((node.get("_meta") or {}).get("title") or "").strip().lower()
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        if class_type == "CLIPTextEncodeLumina2" and title == "positive":
            inputs["user_prompt"] = prompt
        elif class_type == "CLIPTextEncode":
            if title == "positive":
                inputs["text"] = prompt
            elif title == "negative":
                inputs["text"] = negative_prompt
        elif class_type in ("EmptyLatentImage", "EmptySD3LatentImage"):
            inputs["width"] = width
            inputs["height"] = height
        elif class_type == "KSampler" and seed is not None:
            inputs["seed"] = seed

    return workflow


def _queue_prompt(base_url: str, headers: dict, workflow: dict, client_id: str) -> str:
    """Submit a workflow to ComfyUI's /prompt endpoint and return the prompt_id."""
    response = requests.post(
        _resolve_url(base_url, "/prompt"),
        headers=headers,
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    if response.status_code != 200:
        raise http_error(response, "ComfyUI prompt submission")

    data = response.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI response missing prompt_id: {data}")
    return prompt_id


def _wait_for_result(base_url: str, headers: dict, prompt_id: str,
                      timeout: int = POLL_TIMEOUT) -> dict:
    """Poll ComfyUI's /history endpoint until the prompt finishes."""
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(
            _resolve_url(base_url, f"/history/{prompt_id}"),
            headers=headers,
            timeout=30,
        )
        if response.status_code != 200:
            raise http_error(response, "ComfyUI history lookup")

        history = response.json()
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status}")
            outputs = entry.get("outputs")
            if outputs:
                return outputs

        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)

    raise RuntimeError(f"Timed out after {timeout}s waiting for ComfyUI prompt {prompt_id}")


def _extract_image_ref(outputs: dict) -> dict:
    """Find the first image reference (filename/subfolder/type) in the outputs."""
    for node_output in outputs.values():
        images = node_output.get("images") or []
        if images:
            return images[0]
    raise RuntimeError(f"ComfyUI outputs contained no images: {outputs}")


def _generate_image(base_url: str, api_key: str, prompt: str,
                    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                    aspect_ratio: str = "1:1", image_size: str = "1K",
                    output_dir: str = None, filename: str = None,
                    unet_name: str = DEFAULT_UNET, clip_name: str = DEFAULT_CLIP,
                    vae_name: str = DEFAULT_VAE, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                    shift: float = DEFAULT_MODEL_SHIFT, steps: int = DEFAULT_STEPS,
                    cfg: float = DEFAULT_CFG, sampler_name: str = DEFAULT_SAMPLER_NAME,
                    scheduler: str = DEFAULT_SCHEDULER, seed: int = None,
                    workflow_path: str = None) -> str:
    """Generate one image with the ComfyUI backend."""
    width, height = _resolve_size(aspect_ratio, image_size)
    resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    client_id = str(uuid.uuid4())

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if workflow_path:
        workflow = _load_custom_workflow(
            workflow_path, prompt, negative_prompt, width, height, resolved_seed
        )
    else:
        workflow = _default_workflow(
            prompt, negative_prompt, width, height,
            unet_name, clip_name, vae_name, system_prompt, shift,
            steps, cfg, sampler_name, scheduler, resolved_seed,
        )

    print("[ComfyUI Image]")
    print(f"  Base URL:     {base_url}")
    print(f"  UNET:         {unet_name}")
    print(f"  CLIP:         {clip_name}")
    print(f"  VAE:          {vae_name}")
    print(f"  Prompt:       {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"  Aspect Ratio: {aspect_ratio}")
    print(f"  Resolution:   {width}x{height}")
    print(f"  Seed:         {resolved_seed}")
    print()
    print("  [..] Queuing prompt...", end="", flush=True)
    start = time.time()

    prompt_id = _queue_prompt(base_url, headers, workflow, client_id)
    print(f" queued ({prompt_id})")
    print("  [..] Waiting for generation", end="", flush=True)

    outputs = _wait_for_result(base_url, headers, prompt_id)
    elapsed = time.time() - start
    print(f"\n  [DONE] Response received ({elapsed:.1f}s)")

    image_ref = _extract_image_ref(outputs)
    view_params = (
        f"filename={image_ref['filename']}"
        f"&subfolder={image_ref.get('subfolder', '')}"
        f"&type={image_ref.get('type', 'output')}"
    )
    image_url = _resolve_url(base_url, f"/view?{view_params}")

    path = resolve_output_path(prompt, output_dir, filename, ".png")
    return download_image(image_url, path)


def generate(prompt: str, negative_prompt: str = None,
             aspect_ratio: str = "1:1", image_size: str = "1K",
             output_dir: str = None, filename: str = None,
             model: str = None, seed: int = None, max_retries: int = MAX_RETRIES) -> str:
    """Generate an image with retries using the ComfyUI backend."""
    base_url = os.environ.get("COMFYUI_BASE_URL") or DEFAULT_BASE_URL
    api_key = os.environ.get("COMFYUI_API_KEY")
    unet_name = os.environ.get("COMFYUI_UNET") or DEFAULT_UNET
    clip_name = os.environ.get("COMFYUI_CLIP") or DEFAULT_CLIP
    vae_name = os.environ.get("COMFYUI_VAE") or DEFAULT_VAE
    system_prompt = os.environ.get("COMFYUI_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    resolved_negative_prompt = (
        negative_prompt
        if negative_prompt is not None
        else os.environ.get("COMFYUI_NEGATIVE_PROMPT") or DEFAULT_NEGATIVE_PROMPT
    )
    workflow_path = os.environ.get("COMFYUI_WORKFLOW")

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return _generate_image(
                base_url=base_url,
                api_key=api_key,
                prompt=prompt,
                negative_prompt=resolved_negative_prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                output_dir=output_dir,
                filename=filename,
                unet_name=unet_name,
                clip_name=clip_name,
                vae_name=vae_name,
                system_prompt=system_prompt,
                seed=seed,
                workflow_path=workflow_path,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            limited = is_rate_limit_error(exc)
            delay = retry_delay(attempt, rate_limited=limited)
            label = "Rate limit hit" if limited else f"Error: {exc}"
            print(f"\n  [WARN] {label}. Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")