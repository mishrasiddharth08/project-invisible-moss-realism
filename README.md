# MOSS Base Realism for Forge Neo

![Workflow infographic showing the checkpoint, modules, sampling recipe and output](docs/workflow.svg)

![Illustrated Forge Neo UI showing the MOSS Base Realism choice in the Script menu and the Generate button](docs/forge-ui.svg)

*Illustrated preview of the Forge controls. The extension uses the existing Script menu.*

1. Copy this folder into Forge Neo's `extensions` folder. For an existing installation, replace the extension files, then restart Forge.
2. Open **txt2img**. Select the MOSS-Image Base INT8 ConvRot checkpoint from `models/Stable-diffusion`. [Download checkpoint](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors)
3. In **VAE / Text Encoder**, select one MOSS Qwen3.5 9B encoder from `models/text_encoder` and the FLUX.2 VAE from `models/VAE`. [INT8 encoder](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors) · [BF16 encoder](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_bf16.safetensors) · [VAE](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/vae/flux2-vae.safetensors)
4. Enter your prompt. To add a MOSS Base LoRA or LoKr, place its file in `models/Lora`, refresh Forge's **Lora** tab, and click the adapter card. Both use a prompt tag like `<lora:adapter-name:0.8>`. Leave the tag in the prompt when you generate.
5. Choose **Width** and **Height** in multiples of 16. Set **Batch count** for the number of batches and **Batch size** for the number of images in each batch. For four images one at a time, use count **4** and size **1**.
6. In **Script**, choose **MOSS Base Realism**, then click **Generate**. Open an output in **PNG Info** to see the recorded dimensions, adapters and skip layers.

The source recipe used 1184×1776. The preset keeps 25 steps, CFG 3, shift 5, Simple, ER SDE with noise 1.1 and stage 3, and skip layers 22 and 24 on the unconditional pass. Forge's seed of −1 uses the source seed, 855373723910865. The preset runs a single text-to-image pass with an empty negative prompt.

For API use, pass `"script_name": "MOSS Base Realism"` to `POST /sdapi/v1/txt2img` with the checkpoint and modules selected in Forge or supplied through `override_settings`. Forge owns model loading, progress, decoding, saving, and image metadata. Forge and ComfyUI use different noise pipelines, so the same seed can produce a different image.

Tested in an isolated launch of Forge Neo `4f4284b` on an RTX 5090 using the local INT8 encoder. Browser Generate returned one image to the gallery, and the API returned a 1184×1776 PNG with the recipe settings embedded. Forge saved both runs. An SFT request failed before a valid Base request succeeded. BF16 generation and visual parity with ComfyUI remain unverified.

The live Forge UI produced two 768×1024 images plus a preview grid from one prompt with two LoKr tags. An isolated run returned four 384×512 images with batch count 2 and batch size 2. Both LoKrs applied model patches and changed all four images compared with the same prompt and seed without adapters. A sampler trace counted 50 actual skips for each of layers 22 and 24 across the two batches. A separate LoRA generated at 512×384, and Forge rejected an unknown adapter before generation.

Recipe source: [shared ComfyUI workflow](https://pastebin.com/zDg9xMKq), [realism post](https://www.reddit.com/r/StableDiffusion/comments/1xlfsjd/making_mossbase_look_good_for_realism_workflow/), and [showcase](https://www.reddit.com/r/StableDiffusion/comments/1xsa2e2/moss_masterpiece_of_open_source_software/).
