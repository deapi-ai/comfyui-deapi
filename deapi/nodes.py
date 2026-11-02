"""ComfyUI nodes for deAPI (https://deapi.ai)."""

from __future__ import annotations

import logging

from . import media
from .client import DeapiClient, DeapiError

log = logging.getLogger("deAPI")

CATEGORY = "deAPI"
MAX_SEED = 2**31 - 1

API_KEY_INPUT = ("STRING", {
    "default": "",
    "multiline": False,
    "tooltip": "deAPI API key. Leave empty to use the DEAPI_API_KEY environment variable. "
               "A key typed here is saved in the workflow and in the metadata of every image ComfyUI saves. "
               "Get one at https://app.deapi.ai/register",
})
SEED_INPUT = ("INT", {
    "default": -1, "min": -1, "max": MAX_SEED, "control_after_generate": True,
    "tooltip": "-1 = random seed chosen by deAPI (the node re-runs every time).",
})

# ---------------------------------------------------------------- model specs
# Limits mirror GET /api/v2/models. Values outside a model's range are clamped with a log warning.
TXT2IMG_MODELS = {
    "ZImageTurbo_INT8": {"steps": (1, 50, 8), "size": (128, 2048, 16), "negative": True},
    "Flux_2_Klein_4B_BF16": {"steps": (4, 4, 4), "size": (256, 1536, 16), "negative": False},
    "Flux1schnell": {"steps": (1, 10, 4), "size": (256, 2048, 128), "negative": True},
    "ZAnimeDistill_8Step_INT8": {"steps": (4, 16, 8), "size": (128, 2048, 16), "negative": True},
}
EDIT_MODELS = {
    "Flux_2_Klein_4B_BF16": {"steps": (4, 4, 4), "size": (256, 1536, 16), "negative": False,
                             "custom_size": True, "max_images": 3},
    "QwenImageEdit_Plus_NF4": {"steps": (1, 50, 40), "size": (256, 1024, 16), "negative": True,
                               "custom_size": False, "max_images": 1},
}
VIDEO_MODELS = {
    "Ltx2_5_22B_Dist_INT8": {"fps": 24, "frames": (49, 241), "size": (512, 1344, 64), "steps": None,
                             "negative": False},
    "Ltx2_3_22B_Dist_INT8": {"fps": 24, "frames": (49, 241), "size": (512, 1024, 32), "steps": None,
                             "negative": False},
    "MiniMaxH3_33B_Turbo_INT8": {"fps": 24, "frames": (56, 243), "size": (512, 1344, 32), "steps": None,
                                 "negative": False},
    "Ltxv_13B_0_9_8_Distilled_FP8": {"fps": 30, "frames": (30, 120), "size": (256, 768, 32), "steps": 1,
                                     "negative": True},
}
MUSIC_MODELS = ["AceStep_1_5_XL_Turbo_INT8", "AceStep_1_5_Turbo"]
TRANSCRIBE_MODELS = ["WhisperLargeV3", "WhisperLargeV3Ct2"]
OCR_MODELS = ["Nanonets_Ocr_S_F16"]

KOKORO_LANG_BY_PREFIX = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}
KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola", "pf_dora", "pm_alex", "pm_santa",
]


# ---------------------------------------------------------------- helpers
def _clamp(name: str, value: int, lo: int, hi: int, model: str) -> int:
    if value < lo or value > hi:
        new = max(lo, min(hi, value))
        log.warning("deAPI: %s=%s is outside %s's range %s-%s, using %s", name, value, model, lo, hi, new)
        return new
    return value


def _snap(name: str, value: int, lo: int, hi: int, step: int, model: str) -> int:
    value = _clamp(name, value, lo, hi, model)
    snapped = max(lo, min(hi, int(round(value / step)) * step))
    if snapped != value:
        log.warning("deAPI: %s=%s rounded to %s (multiple of %s for %s)", name, value, snapped, step, model)
    return snapped


class _Progress:
    """Maps deAPI job progress (0-100) onto a ComfyUI progress bar and handles Cancel."""

    def __init__(self, jobs: int = 1):
        self.jobs = max(1, jobs)
        self.index = 0
        try:
            import comfy.utils

            self.bar = comfy.utils.ProgressBar(100 * self.jobs)
        except Exception:
            self.bar = None
        try:
            import comfy.model_management as mm

            self.interrupt = mm.throw_exception_if_processing_interrupted
        except Exception:
            self.interrupt = None

    def update(self, pct: float):
        if self.bar is not None:
            self.bar.update_absolute(int(self.index * 100 + max(0.0, min(100.0, pct))), 100 * self.jobs)

    def next(self):
        self.index += 1
        self.update(0)


def _run(client: DeapiClient, path: str, label: str, timeout: float, progress: _Progress, **kwargs) -> dict:
    return client.run(path, label=label, timeout=timeout, on_progress=progress.update,
                      check_interrupt=progress.interrupt, **kwargs)


def _png_part(image, name="image.png"):
    return (name, media.tensor_to_png_bytes(image), "image/png")


def _result_bytes(client: DeapiClient, job: dict, what: str) -> bytes:
    url = job.get("result_url")
    if not url:
        raise DeapiError(f"deAPI finished the {what} job but returned no result_url.")
    return client.download(url)


def _result_text(client: DeapiClient, job: dict) -> str:
    text = job.get("result")
    if text is None and job.get("result_url"):
        text = client.download(job["result_url"]).decode("utf-8", errors="replace")
    return text or ""


def _random_seed_changed(seed: int):
    return float("nan") if seed == -1 else seed


# ---------------------------------------------------------------- nodes
class DeapiTextToImage:
    DESCRIPTION = "Generate an image from a text prompt with deAPI."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a red fox in fresh snow, golden hour, 35mm photo"}),
                "model": (list(TXT2IMG_MODELS), {"default": "ZImageTurbo_INT8"}),
                "width": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50,
                                  "tooltip": "Klein: 4. Flux1schnell: 1-10 (4). Z-Image: 1-50 (8). Z-Anime: 4-16 (8)."}),
                "seed": SEED_INPUT,
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "Ignored by FLUX.2 Klein."}),
                "api_key": API_KEY_INPUT,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return _random_seed_changed(seed)

    def generate(self, prompt, model, width, height, steps, seed, negative_prompt="", api_key=""):
        spec = TXT2IMG_MODELS[model]
        lo, hi, step = spec["size"]
        body = {
            "prompt": prompt,
            "model": model,
            "width": _snap("width", width, lo, hi, step, model),
            "height": _snap("height", height, lo, hi, step, model),
            "steps": _clamp("steps", steps, spec["steps"][0], spec["steps"][1], model),
            "seed": seed,
        }
        if negative_prompt.strip() and spec["negative"]:
            body["negative_prompt"] = negative_prompt
        client = DeapiClient(api_key)
        progress = _Progress()
        job = _run(client, "/images/generations", "text-to-image", 600, progress, json=body)
        img = media.bytes_to_pil(_result_bytes(client, job, "text-to-image"))
        return (media.pil_to_image_tensor(img),)


class DeapiImageEdit:
    DESCRIPTION = ("Edit an image with a text instruction. FLUX.2 Klein is fast (seconds) and accepts up to "
                   "3 reference images; Qwen Image Edit Plus is slower (minutes) but follows instructions closely.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "make it look like a watercolor painting"}),
                "model": (list(EDIT_MODELS), {"default": "Flux_2_Klein_4B_BF16"}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 50,
                                  "tooltip": "0 = model default (Klein 4, Qwen 40). Fewer Qwen steps = faster."}),
                "seed": SEED_INPUT,
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Extra reference image (FLUX.2 Klein only)."}),
                "image_3": ("IMAGE", {"tooltip": "Extra reference image (FLUX.2 Klein only)."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Qwen only."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 1536, "step": 16,
                                  "tooltip": "Output width, 0 = same as input (Klein only)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 1536, "step": 16,
                                   "tooltip": "Output height, 0 = same as input (Klein only)."}),
                "api_key": API_KEY_INPUT,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return _random_seed_changed(seed)

    def edit(self, image, prompt, model, steps, seed, image_2=None, image_3=None, negative_prompt="",
             width=0, height=0, api_key=""):
        spec = EDIT_MODELS[model]
        smin, smax, sdef = spec["steps"]
        data = {
            "prompt": prompt,
            "model": model,
            "steps": sdef if steps <= 0 else _clamp("steps", steps, smin, smax, model),
            "seed": seed,
        }
        if negative_prompt.strip() and spec["negative"]:
            data["negative_prompt"] = negative_prompt
        if spec["custom_size"] and width > 0 and height > 0:
            lo, hi, step = spec["size"]
            data["width"] = _snap("width", width, lo, hi, step, model)
            data["height"] = _snap("height", height, lo, hi, step, model)
        refs = [r for r in (image_2, image_3) if r is not None]
        if refs and spec["max_images"] == 1:
            log.warning("deAPI: %s accepts a single image, extra reference images are ignored", model)
            refs = []

        client = DeapiClient(api_key)
        items = list(media.batch_items(image))
        progress = _Progress(len(items))
        timeout = 1800 if model.startswith("Qwen") else 600
        outputs = []
        for i, item in enumerate(items):
            if refs:
                files = [("images[]", _png_part(item, "image_1.png"))]
                files += [("images[]", _png_part(r, f"image_{n + 2}.png")) for n, r in enumerate(refs)]
            else:
                files = {"image": _png_part(item)}
            job = _run(client, "/images/edits", "image edit", timeout, progress, data=data, files=files)
            outputs.append(media.pil_to_image_tensor(media.bytes_to_pil(_result_bytes(client, job, "image edit"))))
            if i + 1 < len(items):
                progress.next()
        return (media.cat_images(outputs),)


class DeapiRemoveBackground:
    DESCRIPTION = "Remove the background of an image (Ben2). Returns the RGB image and a foreground mask (1 = subject)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",), "model": (["Ben2"],)},
            "optional": {"api_key": API_KEY_INPUT},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(self, image, model, api_key=""):
        client = DeapiClient(api_key)
        items = list(media.batch_items(image))
        progress = _Progress(len(items))
        images, masks = [], []
        for i, item in enumerate(items):
            job = _run(client, "/images/background-removals", "background removal", 600, progress,
                       data={"model": model}, files={"image": _png_part(item)})
            rgb, alpha = media.pil_to_rgb_and_alpha(media.bytes_to_pil(_result_bytes(client, job, "background removal")))
            images.append(rgb)
            masks.append(alpha)
            if i + 1 < len(items):
                progress.next()
        return (media.cat_images(images), media.cat_images(masks))


class DeapiUpscale:
    DESCRIPTION = "Upscale an image 4x (RealESRGAN x4). Input sides must be 128-2048 px."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",), "model": (["RealESRGAN_x4"],)},
            "optional": {"api_key": API_KEY_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, image, model, api_key=""):
        client = DeapiClient(api_key)
        items = list(media.batch_items(image))
        progress = _Progress(len(items))
        outputs = []
        for i, item in enumerate(items):
            job = _run(client, "/images/upscales", "upscale", 600, progress,
                       data={"model": model}, files={"image": _png_part(item)})
            outputs.append(media.pil_to_image_tensor(media.bytes_to_pil(_result_bytes(client, job, "upscale"))))
            if i + 1 < len(items):
                progress.next()
        return (media.cat_images(outputs),)


def _video_inputs(default_prompt: str):
    return {
        "prompt": ("STRING", {"multiline": True, "default": default_prompt}),
        "model": (list(VIDEO_MODELS), {"default": "Ltx2_5_22B_Dist_INT8"}),
        "width": ("INT", {"default": 768, "min": 256, "max": 1344, "step": 32}),
        "height": ("INT", {"default": 768, "min": 256, "max": 1344, "step": 32}),
        "frames": ("INT", {"default": 120, "min": 30, "max": 243,
                           "tooltip": "LTX-2.x: 49-241 @ 24 fps. MiniMax: 56-243 @ 24 fps. LTX 13B: 30-120 @ 30 fps."}),
        "seed": SEED_INPUT,
    }


def _video_body(prompt, model, width, height, frames, seed, negative_prompt):
    spec = VIDEO_MODELS[model]
    lo, hi, step = spec["size"]
    body = {
        "prompt": prompt,
        "model": model,
        "width": _snap("width", width, lo, hi, step, model),
        "height": _snap("height", height, lo, hi, step, model),
        "frames": _clamp("frames", frames, spec["frames"][0], spec["frames"][1], model),
        "fps": spec["fps"],
        "seed": seed,
    }
    if spec["steps"]:
        body["steps"] = spec["steps"]
    if negative_prompt.strip() and spec["negative"]:
        body["negative_prompt"] = negative_prompt
    return body


def _video_outputs(client, job, prefix):
    path = media.save_bytes(_result_bytes(client, job, "video"), prefix, "mp4")
    frames, _fps = media.video_frames(path)
    return (media.video_object(path), frames, path)


VIDEO_RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
VIDEO_RETURN_NAMES = ("video", "frames", "file_path")


class DeapiTextToVideo:
    DESCRIPTION = "Generate a video clip from a text prompt. Takes about 1-3 minutes. The MP4 is also saved to output/deapi/."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _video_inputs("a fox running through fresh snow, cinematic tracking shot"),
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "LTX 13B only."}),
                "api_key": API_KEY_INPUT,
            },
        }

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return _random_seed_changed(seed)

    def generate(self, prompt, model, width, height, frames, seed, negative_prompt="", api_key=""):
        client = DeapiClient(api_key)
        body = _video_body(prompt, model, width, height, frames, seed, negative_prompt)
        job = _run(client, "/videos/generations", "text-to-video", 1800, _Progress(), json=body)
        return _video_outputs(client, job, "t2v")


class DeapiImageToVideo:
    DESCRIPTION = ("Animate an image into a video clip (first frame, optional last frame). "
                   "Takes about 1-3 minutes. The MP4 is also saved to output/deapi/.")

    @classmethod
    def INPUT_TYPES(cls):
        req = {"first_frame": ("IMAGE",)}
        req.update(_video_inputs("the camera slowly pushes in, gentle natural motion"))
        return {
            "required": req,
            "optional": {
                "last_frame": ("IMAGE", {"tooltip": "Optional image the clip should end on."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "LTX 13B only."}),
                "api_key": API_KEY_INPUT,
            },
        }

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "animate"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return _random_seed_changed(seed)

    def animate(self, first_frame, prompt, model, width, height, frames, seed, last_frame=None,
                negative_prompt="", api_key=""):
        client = DeapiClient(api_key)
        data = _video_body(prompt, model, width, height, frames, seed, negative_prompt)
        files = {"first_frame_image": _png_part(first_frame, "first_frame.png")}
        if last_frame is not None:
            files["last_frame_image"] = _png_part(last_frame, "last_frame.png")
        job = _run(client, "/videos/animations", "image-to-video", 1800, _Progress(), data=data, files=files)
        return _video_outputs(client, job, "i2v")


class DeapiTextToSpeech:
    DESCRIPTION = "Turn text into speech with Kokoro (41 voices, 7 languages; the language follows the voice)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello! This voice was generated with deAPI."}),
                "voice": (KOKORO_VOICES, {"default": "af_heart",
                                          "tooltip": "Prefix: a=US English, b=UK English, e=Spanish, f=French, "
                                                     "h=Hindi, i=Italian, p=Brazilian Portuguese; f/m = female/male."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            },
            "optional": {"api_key": API_KEY_INPUT},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speak"
    CATEGORY = CATEGORY

    def speak(self, text, voice, speed, api_key=""):
        if len(text.strip()) < 3:
            raise DeapiError("Text to speech needs at least 3 characters of text.")
        client = DeapiClient(api_key)
        data = {
            "text": text,
            "model": "Kokoro",
            "voice": voice,
            "lang": KOKORO_LANG_BY_PREFIX.get(voice[0], "en-us"),
            "speed": speed,
            "format": "mp3",
            "sample_rate": 24000,
        }
        job = _run(client, "/audio/speech", "text-to-speech", 600, _Progress(), data=data)
        return (media.decode_audio(_result_bytes(client, job, "text-to-speech")),)


class DeapiTextToMusic:
    DESCRIPTION = "Generate music (ACE-Step 1.5). Use [Instrumental] as lyrics for music without vocals."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"multiline": True, "default": "calm lo-fi hip hop, warm piano, vinyl crackle",
                                       "tooltip": "Style description, 3-300 characters."}),
                "lyrics": ("STRING", {"multiline": True, "default": "[Instrumental]"}),
                "model": (MUSIC_MODELS, {"default": MUSIC_MODELS[0]}),
                "duration": ("INT", {"default": 30, "min": 10, "max": 300, "tooltip": "Seconds."}),
                "seed": SEED_INPUT,
            },
            "optional": {
                "bpm": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = let the model decide (50-200)."}),
                "api_key": API_KEY_INPUT,
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "compose"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return _random_seed_changed(seed)

    def compose(self, caption, lyrics, model, duration, seed, bpm=0, api_key=""):
        client = DeapiClient(api_key)
        data = {
            "caption": caption,
            "lyrics": lyrics or "[Instrumental]",
            "model": model,
            "duration": duration,
            "inference_steps": 8,
            "guidance_scale": 1,
            "seed": seed,
            "format": "mp3",
        }
        if bpm:
            data["bpm"] = _clamp("bpm", bpm, 50, 200, model)
        job = _run(client, "/audio/music", "text-to-music", 900, _Progress(), data=data)
        return (media.decode_audio(_result_bytes(client, job, "text-to-music")),)


class DeapiTranscribe:
    DESCRIPTION = ("Transcribe speech to text with Whisper. Connect an AUDIO input, or paste a media URL "
                   "(YouTube, X, Twitch, Kick, or a direct audio/video link).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (TRANSCRIBE_MODELS, {"default": "WhisperLargeV3"}),
                "include_timestamps": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "source_url": ("STRING", {"default": "", "tooltip": "Used when no audio is connected."}),
                "language": ("STRING", {"default": "", "tooltip": "Optional ISO code, e.g. en. Empty = auto-detect."}),
                "api_key": API_KEY_INPUT,
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def transcribe(self, model, include_timestamps, audio=None, source_url="", language="", api_key=""):
        data = {"model": model, "include_ts": include_timestamps, "return_result_in_response": True}
        if language.strip():
            data["lang"] = language.strip()
        files = None
        if audio is not None:
            files = {"source_file": ("audio.wav", media.audio_to_wav_bytes(audio), "audio/wav")}
        elif source_url.strip():
            data["source_url"] = source_url.strip()
        else:
            raise DeapiError("Connect an AUDIO input or fill in source_url.")
        client = DeapiClient(api_key)
        job = _run(client, "/audio/transcriptions", "transcription", 1800, _Progress(), data=data, files=files)
        text = _result_text(client, job)
        return {"ui": {"text": [text]}, "result": (text,)}


class DeapiImageToText:
    DESCRIPTION = "Extract text from an image (OCR) with Nanonets OCR."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",), "model": (OCR_MODELS,)},
            "optional": {"api_key": API_KEY_INPUT},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ocr"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def ocr(self, image, model, api_key=""):
        client = DeapiClient(api_key)
        items = list(media.batch_items(image))
        progress = _Progress(len(items))
        texts = []
        for i, item in enumerate(items):
            job = _run(client, "/images/ocr", "OCR", 600, progress,
                       data={"model": model, "return_result_in_response": True}, files={"image": _png_part(item)})
            texts.append(_result_text(client, job))
            if i + 1 < len(items):
                progress.next()
        text = "\n\n".join(texts)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "deAPITextToImage": DeapiTextToImage,
    "deAPIImageEdit": DeapiImageEdit,
    "deAPIRemoveBackground": DeapiRemoveBackground,
    "deAPIUpscale": DeapiUpscale,
    "deAPITextToVideo": DeapiTextToVideo,
    "deAPIImageToVideo": DeapiImageToVideo,
    "deAPITextToSpeech": DeapiTextToSpeech,
    "deAPITextToMusic": DeapiTextToMusic,
    "deAPITranscribe": DeapiTranscribe,
    "deAPIImageToText": DeapiImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "deAPITextToImage": "deAPI Text to Image",
    "deAPIImageEdit": "deAPI Image Edit",
    "deAPIRemoveBackground": "deAPI Remove Background",
    "deAPIUpscale": "deAPI Upscale Image",
    "deAPITextToVideo": "deAPI Text to Video",
    "deAPIImageToVideo": "deAPI Image to Video",
    "deAPITextToSpeech": "deAPI Text to Speech",
    "deAPITextToMusic": "deAPI Text to Music",
    "deAPITranscribe": "deAPI Transcribe Audio",
    "deAPIImageToText": "deAPI Image to Text (OCR)",
}
