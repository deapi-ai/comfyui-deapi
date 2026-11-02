"""Conversions between ComfyUI types (IMAGE / MASK / AUDIO / VIDEO) and files."""

from __future__ import annotations

import io
import os
import uuid
import wave

import numpy as np
import torch
from PIL import Image, ImageOps


# ------------------------------------------------------------------ images
def tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    """One ComfyUI image [H,W,C] (or [1,H,W,C]) float 0..1 -> PNG bytes."""
    if image.dim() == 4:
        image = image[0]
    arr = (image.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def batch_items(images: torch.Tensor):
    """Yield each image of a ComfyUI IMAGE batch."""
    if images.dim() == 3:
        yield images
        return
    for i in range(images.shape[0]):
        yield images[i]


def bytes_to_pil(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img


def pil_to_image_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI IMAGE [1,H,W,3]."""
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def pil_to_rgb_and_alpha(img: Image.Image):
    """PIL (possibly RGBA) -> (IMAGE [1,H,W,3], MASK [1,H,W] where 1 = opaque/foreground)."""
    if img.mode not in ("RGBA", "LA") and "transparency" not in img.info:
        image = pil_to_image_tensor(img)
        mask = torch.ones(image.shape[1:3], dtype=torch.float32)[None, ...]
        return image, mask
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    image = torch.from_numpy(arr[..., :3])[None, ...]
    mask = torch.from_numpy(arr[..., 3])[None, ...]
    return image, mask


def cat_images(tensors):
    """Concatenate IMAGE (or MASK) tensors; resize to the first one's size if needed."""
    if len(tensors) == 1:
        return tensors[0]
    target = tensors[0].shape[1:3]
    out = [tensors[0]]
    for t in tensors[1:]:
        if t.shape[1:3] != target:
            if t.dim() == 4:
                t = torch.nn.functional.interpolate(t.movedim(-1, 1), size=tuple(target), mode="bilinear",
                                                    align_corners=False).movedim(1, -1)
            else:
                t = torch.nn.functional.interpolate(t[:, None], size=tuple(target), mode="bilinear",
                                                    align_corners=False)[:, 0]
        out.append(t)
    return torch.cat(out, dim=0)


# ------------------------------------------------------------------ audio
def decode_audio(data: bytes) -> dict:
    """Encoded audio bytes (mp3/wav/...) -> ComfyUI AUDIO dict."""
    import av  # shipped with ComfyUI

    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            raise ValueError("The deAPI result has no audio stream.")
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels or 1
        chunks = []
        for frame in container.decode(stream):
            buf = frame.to_ndarray()
            if buf.ndim == 1:
                buf = buf[None, :]
            if buf.shape[0] != channels:  # packed/interleaved layout
                buf = buf.reshape(-1, channels).T
            chunks.append(buf)
    if not chunks:
        raise ValueError("The deAPI audio result is empty.")
    wav = np.concatenate(chunks, axis=1)
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483648.0
    else:
        wav = wav.astype(np.float32)
    return {"waveform": torch.from_numpy(np.ascontiguousarray(wav))[None, ...], "sample_rate": int(sample_rate)}


def audio_to_wav_bytes(audio: dict) -> bytes:
    """ComfyUI AUDIO dict -> 16-bit mono WAV bytes (first item of the batch)."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform[None, :]
    mono = waveform.detach().cpu().float().mean(dim=0).clamp(-1, 1).numpy()
    pcm = (mono * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ------------------------------------------------------------------ video
def output_dir() -> str:
    try:
        import folder_paths

        base = folder_paths.get_output_directory()
    except Exception:  # running outside ComfyUI
        base = os.path.join(os.getcwd(), "output")
    path = os.path.join(base, "deapi")
    os.makedirs(path, exist_ok=True)
    return path


def save_bytes(data: bytes, prefix: str, ext: str) -> str:
    path = os.path.join(output_dir(), f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def video_object(path: str):
    """Wrap a video file as a ComfyUI VIDEO object (None on old ComfyUI without comfy_api)."""
    try:
        from comfy_api.input_impl import VideoFromFile
    except Exception:
        try:
            from comfy_api.latest import InputImpl

            VideoFromFile = InputImpl.VideoFromFile
        except Exception:
            return None
    return VideoFromFile(path)


def video_frames(path: str):
    """Decode a video file to (IMAGE [N,H,W,3], fps)."""
    import av

    frames = []
    fps = 0.0
    with av.open(path) as container:
        stream = container.streams.video[0]
        if stream.average_rate:
            fps = float(stream.average_rate)
        for frame in container.decode(stream):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise ValueError("The deAPI video result has no frames.")
    return torch.stack(frames).float() / 255.0, fps
