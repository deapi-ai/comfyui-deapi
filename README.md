<p align="center">
  <img src="assets/icon.png" width="96" alt="deAPI">
</p>

# deAPI nodes for ComfyUI

Run image, video, speech, music and transcription models from [deAPI](https://deapi.ai) inside ComfyUI. Jobs run on deAPI's cloud GPUs, so you don't need a local GPU or model downloads. You pay per job at deAPI's prices.

All nodes use standard ComfyUI types (`IMAGE`, `MASK`, `VIDEO`, `AUDIO`, `STRING`), so they connect to core nodes such as Load Image, Save Image, Save Video and Save Audio, and to any other custom nodes.

## Install

**ComfyUI Manager / Registry (recommended)**

Search for **deAPI** in ComfyUI Manager, or run:

```bash
comfy node install comfyui-deapi
```

**Manual**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/deapi-ai/comfyui-deapi
pip install -r comfyui-deapi/requirements.txt   # only needs `requests`, which ComfyUI already ships with
```

Restart ComfyUI. The nodes appear under the **deAPI** category, and you can find them by searching "deAPI".

## API key

1. Create a free account at **https://app.deapi.ai/register** and copy your API key from the dashboard.
2. Pick one of these:
   - **Environment variable (recommended):** set `DEAPI_API_KEY` before starting ComfyUI.
     - Windows (PowerShell): `$env:DEAPI_API_KEY="your-key"; python main.py`
     - macOS / Linux: `DEAPI_API_KEY=your-key python main.py`
   - **Node input:** paste the key into the optional `api_key` field on a node. Keys typed there are saved inside the workflow, so don't share those workflows.

## Nodes

| Node | Inputs | Outputs | Models |
|---|---|---|---|
| **deAPI Text to Image** | prompt, model, width, height, steps, seed, *negative_prompt* | `IMAGE` | Z-Image-Turbo, FLUX.2 Klein 4B, FLUX.1 Schnell, Z-Anime Distill |
| **deAPI Image Edit** | image, prompt, model, steps, seed, *image_2, image_3, negative_prompt, width, height* | `IMAGE` | FLUX.2 Klein 4B (fast, up to 3 reference images), Qwen Image Edit Plus (slower, several minutes) |
| **deAPI Remove Background** | image | `IMAGE`, `MASK` | Ben2 |
| **deAPI Upscale Image** | image | `IMAGE` (4x) | RealESRGAN x4 |
| **deAPI Text to Video** | prompt, model, width, height, frames, seed, *negative_prompt* | `VIDEO`, `IMAGE` (frames), `STRING` (file path) | LTX-2.5 22B, LTX-2.3 22B, MiniMax H3 33B, LTX-Video 13B |
| **deAPI Image to Video** | first_frame, prompt, model, width, height, frames, seed, *last_frame, negative_prompt* | `VIDEO`, `IMAGE` (frames), `STRING` (file path) | same as Text to Video |
| **deAPI Text to Speech** | text, voice, speed | `AUDIO` | Kokoro (41 voices in US/UK English, Spanish, French, Hindi, Italian and Brazilian Portuguese) |
| **deAPI Text to Music** | caption, lyrics, model, duration, seed, *bpm* | `AUDIO` | ACE-Step 1.5 XL Turbo, ACE-Step 1.5 Turbo |
| **deAPI Transcribe Audio** | model, include_timestamps, *audio, source_url, language* | `STRING` | Whisper Large V3 |
| **deAPI Image to Text (OCR)** | image | `STRING` | Nanonets OCR |

Inputs in *italics* are optional. Every node also has an optional `api_key` input.

How the nodes behave:

- **Seed:** `-1` lets deAPI pick a random seed, and the node runs again on every queue. Any other value is reproducible.
- **Limits:** each model has its own size, step and frame limits. If a value is out of range, the node clamps it to the nearest allowed value and logs a warning in the console. Video fps is fixed per model: 24 fps for LTX-2.x and MiniMax, 30 fps for LTX-Video 13B.
- **Video outputs:** video nodes return a native `VIDEO`, so you can connect **Save Video** or **Preview Video**. They also return the decoded frames as `IMAGE` and the path of the MP4, which is saved in `ComfyUI/output/deapi/`.
- **Remove Background mask:** the `MASK` output is 1 on the subject. To get a transparent PNG, connect it through **InvertMask** into **Join Image with Alpha**, then **Save Image**. The example workflow shows this.
- **Transcribe Audio:** accepts an `AUDIO` input or a `source_url` (YouTube, X, Twitch, Kick, or a direct media link). The text appears on the node and is also returned as a `STRING`.
- **Batches:** image nodes process every image in an input batch.
- **Progress and cancelling:** progress shows on the node, and status lines are printed to the console. Cancel stops waiting for a job, but a job that was already submitted may still finish (and be billed) on deAPI.

## Example workflows

The [`example_workflows/`](example_workflows) folder has ready-made workflows. Drag one onto the ComfyUI canvas to load it, or open it from the **Templates** browser under *Custom Nodes*.

- `deapi_text_to_image.json`: text to image, then Save Image.
- `deapi_image_edit_and_remove_background.json`: edit an image, cut out the subject as a transparent PNG, and upscale it 4x.
- `deapi_image_to_video.json`: generate a still, animate it, then Save Video.
- `deapi_text_to_video.json`: text to video, then Save Video.
- `deapi_speech_music_transcribe.json`: text to speech (saved and transcribed back to text) and text to music.

## Typical timings

These are wall-clock times measured from ComfyUI, and they vary with load.

| Task | Time |
|---|---|
| Text to image (768x768) | 10-20 s |
| Image edit, FLUX.2 Klein | 5-25 s |
| Remove background | 5-10 s |
| Upscale 4x | 20-50 s |
| Text to speech, transcription, OCR | 5-15 s |
| Image edit, Qwen Image Edit Plus | several minutes |
| Text or image to video | 1-3 min |

## Errors

Errors appear in the ComfyUI error dialog, and they include the HTTP status and deAPI's own message. These are the common ones:

- **`deAPI HTTP 401 [invalid_key]`**: the key is missing or wrong. Check `DEAPI_API_KEY` or the node's `api_key` field.
- **`deAPI HTTP 422: ... balance ...`**: your balance is too low. Top up at https://app.deapi.ai.
- **`deAPI HTTP 422: The <field> field is ...`**: a parameter was rejected. The message names the field.
- **`deAPI HTTP 429`**: you hit the rate limit. The node retries automatically a few times before giving up.
- **`timed out after N s`**: the job took longer than the node waits. The message includes the request id, so you can look the job up later.

## Configuration

| Environment variable | Purpose |
|---|---|
| `DEAPI_API_KEY` | API key used when a node's `api_key` field is empty |
| `DEAPI_BASE_URL` | Override the API base URL (default `https://api.deapi.ai/api/v2`) |

## Links

- deAPI: https://deapi.ai
- API docs: https://docs.deapi.ai
- Get an API key: https://app.deapi.ai/register

## License

MIT © deAPI Inc.
