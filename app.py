# flake8: noqa
import os
import threading
import datetime
import logging
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import PngImagePlugin
from diffusers import DiffusionPipeline


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("lcm_app")

app = FastAPI(title="LCM Text2Image")


# Environment configuration with sensible defaults
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data")
MODEL_ID = os.getenv("MODEL_ID", "SimianLuo/LCM_Dreamshaper_v7")
LCM_REVISION = os.getenv("LCM_REVISION", "fb9c5d")
LCM_CUSTOM_PIPELINE = os.getenv("LCM_CUSTOM_PIPELINE", "latent_consistency_txt2img")
LCM_CUSTOM_REVISION = os.getenv("LCM_CUSTOM_REVISION", "main")
DEFAULT_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "8"))
DEFAULT_GUIDANCE = float(os.getenv("GUIDANCE_SCALE", "30.0"))
DEFAULT_LCM_ORIGIN_STEPS = int(os.getenv("LCM_ORIGIN_STEPS", "8"))
SAFETY_CHECKER = os.getenv("SAFETY_CHECKER", "disabled").lower()  # "disabled" or "default"
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1") in {"1", "true", "yes"}


os.makedirs(OUTPUT_DIR, exist_ok=True)


# Serve generated files for convenience
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


_pipeline_lock = threading.Lock()
_pipeline: Optional[DiffusionPipeline] = None


def _create_pipeline() -> DiffusionPipeline:
    safety_arg = {} if SAFETY_CHECKER != "disabled" else {"safety_checker": None}

    # Use float16 on CUDA, float32 on CPU
    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    logger.info("Initializing diffusion pipeline: model_id=%s, device=%s, dtype=%s, safety=%s",
                MODEL_ID, "cuda" if use_cuda else "cpu", str(torch_dtype), "enabled" if SAFETY_CHECKER != "disabled" else "disabled")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        custom_pipeline=LCM_CUSTOM_PIPELINE,
        custom_revision=LCM_CUSTOM_REVISION,
        revision=LCM_REVISION,
        torch_dtype=torch_dtype,
        **safety_arg,
    )

    device = "cuda" if use_cuda else "cpu"
    pipe = pipe.to(device)
    logger.info("Pipeline ready on device=%s", device)
    return pipe


def _get_pipeline() -> DiffusionPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = _create_pipeline()
    return _pipeline


def _save_image(image, filename: str, metadata: dict) -> None:
    meta_tuples = [(k, str(v)) for k, v in metadata.items()]
    png_info = PngImagePlugin.PngInfo()
    for k, v in meta_tuples:
        png_info.add_text(k, v)
    image.save(filename, pnginfo=png_info)


def _generate_filename(prompt: str, timestamp: str, index: int) -> str:
    snippet = "_".join(prompt.split()[:3]) or "image"
    return f"{timestamp}_{snippet}_{index}.png"


@app.on_event("startup")
def _on_startup():
    # Optionally preload model to avoid first-request latency
    if PRELOAD_MODEL:
        logger.info("Preloading model on startup")
        _get_pipeline()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


INDEX_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>LCM Text2Image</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 12px; }
    h1 { font-size: 20px; }
    label { display: block; margin: 12px 0 6px; font-weight: 600; }
    input, textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
    button { margin-top: 16px; padding: 10px 16px; background: #111827; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button:disabled { opacity: .6; cursor: not-allowed; }
    .row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-top: 24px; }
    .img-card { border: 1px solid #e5e7eb; padding: 8px; border-radius: 8px; }
    .small { color: #6b7280; font-size: 12px; }
    .note { background: #f9fafb; padding: 8px; border-radius: 6px; border: 1px solid #e5e7eb; }
  </style>
  <script>
    async function generate(evt) {
      evt.preventDefault();
      const form = document.getElementById('genForm');
      const data = new FormData(form);
      const btn = document.getElementById('go');
      btn.disabled = true; btn.innerText = 'Generating...';
      try {
        const res = await fetch('/api/generate', { method: 'POST', body: data });
        if (!res.ok) throw new Error('Request failed');
        const json = await res.json();
        const grid = document.getElementById('grid');
        grid.innerHTML = '';
        for (const f of json.files) {
          const card = document.createElement('div');
          card.className = 'img-card';
          const img = document.createElement('img');
          img.src = f.url; img.style.width = '100%'; img.loading = 'lazy';
          const cap = document.createElement('div');
          cap.className = 'small'; cap.innerText = f.name;
          card.appendChild(img); card.appendChild(cap); grid.appendChild(card);
        }
      } catch (e) {
        alert('Error: ' + e.message);
      } finally {
        btn.disabled = false; btn.innerText = 'Generate';
      }
    }
  </script>
  </head>
  <body>
    <h1>LCM Text2Image</h1>
    <form id=\"genForm\" onsubmit=\"generate(event)\">
      <label>Prompt</label>
      <textarea name=\"prompt\" rows=\"3\" placeholder=\"A scenic watercolor landscape with mountains...\" required></textarea>
      <div class=\"row\">
        <div>
          <label>Images</label>
          <input type=\"number\" name=\"num_images\" min=\"1\" max=\"8\" value=\"1\" />
        </div>
        <div>
          <label>Steps</label>
          <input type=\"number\" name=\"num_inference_steps\" min=\"1\" max=\"20\" value=\"{DEFAULT_STEPS}\" />
        </div>
        <div>
          <label>Guidance</label>
          <input type=\"number\" step=\"0.5\" name=\"guidance_scale\" value=\"{DEFAULT_GUIDANCE}\" />
        </div>
      </div>
      <div class=\"row\">
        <div>
          <label>LCM Origin Steps</label>
          <input type=\"number\" name=\"lcm_origin_steps\" min=\"1\" max=\"20\" value=\"{DEFAULT_LCM_ORIGIN_STEPS}\" />
        </div>
      </div>
      <button id=\"go\" type=\"submit\">Generate</button>
    </form>
    <div class=\"note small\" style=\"margin-top:12px\">Outputs are saved to /outputs (container) and mapped host directory.</div>
    <div id=\"grid\" class=\"grid\"></div>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    html = (
        INDEX_HTML
        .replace("{DEFAULT_STEPS}", str(DEFAULT_STEPS))
        .replace("{DEFAULT_GUIDANCE}", str(DEFAULT_GUIDANCE))
        .replace("{DEFAULT_LCM_ORIGIN_STEPS}", str(DEFAULT_LCM_ORIGIN_STEPS))
    )
    return HTMLResponse(html)


@app.post("/api/generate")
def api_generate(
    prompt: str = Form(...),
    num_images: int = Form(1),
    num_inference_steps: int = Form(None),
    guidance_scale: float = Form(None),
    lcm_origin_steps: int = Form(None),
):
    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt is required")

    steps = num_inference_steps or DEFAULT_STEPS
    guidance = guidance_scale or DEFAULT_GUIDANCE
    origin_steps = lcm_origin_steps or DEFAULT_LCM_ORIGIN_STEPS

    logger.info("Generation request: prompt=%r, num_images=%s, steps=%s, guidance=%s, origin_steps=%s",
                prompt[:80], num_images, steps, guidance, origin_steps)
    try:
        pipe = _get_pipeline()
        images = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            lcm_origin_steps=origin_steps,
            num_images_per_prompt=max(1, num_images),
            output_type="pil",
        ).images
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    ts = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    metadata = {"prompt": prompt, "num_steps": steps, "guidance": guidance, "origin_steps": origin_steps}
    files: List[dict] = []

    for idx, image in enumerate(images[: max(1, num_images)]):
        filename = _generate_filename(prompt, ts, idx)
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            _save_image(image, filepath, metadata)
        except Exception as e:
            logger.exception("Failed to save image: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
        files.append({
            "name": filename,
            "path": filepath,
            "url": f"/outputs/{filename}",
        })
    logger.info("Generated %d images", len(files))
    return JSONResponse({"files": files})


