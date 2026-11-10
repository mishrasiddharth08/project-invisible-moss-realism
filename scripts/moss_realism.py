from functools import lru_cache, partial
from pathlib import Path

from k_diffusion import sampling
from modules import extra_networks, hashes, processing, scripts, sd_models, sd_samplers, sd_schedulers, shared
from modules_forge import main_entry


TITLE = "MOSS Base Realism"
BASE = "f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f"
VAE = "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"
ENCODERS = {
    "7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53": "BF16",
    "32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a": "INT8",
}
OPTIONS = {
    "slg_enable": True,
    "slg_single_layers": "22, 24",
    "slg_start_percent": 0.0,
    "slg_end_percent": 1.0,
    "always_discard_next_to_last_sigma": False,
    "skip_early_cond": 0.0,
}


@lru_cache(maxsize=16)
def _hash_file(path, size, modified_ns, changed_ns):
    return hashes.calculate_sha256_real(path)


def fingerprint(filename):
    try:
        path = Path(filename).resolve(strict=True)
        stat = path.stat()
        return _hash_file(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError as error:
        raise ValueError(f"Forge could not read {Path(filename).name}. Refresh the model list and select the file again.") from error


def _skip_incompatible(p, checkpoint):
    name = Path(getattr(checkpoint, "filename", "") or "selected checkpoint").name
    message = (
        f"MOSS Base Realism skipped: {name} is not the verified MOSS-Image Base INT8 ConvRot checkpoint. "
        "Normal Forge generation will continue with the selected model."
    )
    p.extra_generation_params["MOSS Base Realism"] = "Skipped: incompatible checkpoint"
    try:
        import gradio as gr
        gr.Warning(message)
    except Exception:
        pass
    print(f"[MOSS Base Realism] {message}")
    # Native Generate falls through on None; Forge's selectable-script API
    # expects a Processed object and has no corresponding fallback.
    if getattr(p, "is_api", False):
        return processing.process_images(p)
    return None


class MossRealism(scripts.Script):
    def title(self):
        return TITLE

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        return []

    def run(self, p, *args):
        override = getattr(p, "override_settings", None) or {}
        checkpoint_name = override.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint)
        checkpoint = sd_models.checkpoint_aliases.get(checkpoint_name)
        if checkpoint is None or fingerprint(checkpoint.filename) != BASE:
            return _skip_incompatible(p, checkpoint)

        if (
            "ER SDE" not in sd_samplers.all_samplers_map
            or "Simple" not in sd_schedulers.schedulers_map
            or not OPTIONS.keys() <= shared.opts.data_labels.keys()
        ):
            raise RuntimeError("MOSS Base Realism requires Forge Neo with ER SDE, the Simple scheduler and skip-layer guidance.")
        if getattr(p, "txt2img_upscale", False):
            raise ValueError("MOSS Base Realism is a single-pass preset. Use Generate for a new image.")

        if any(dimension < 64 or dimension % 16 for dimension in (p.width, p.height)):
            raise ValueError("Set Width and Height to multiples of 16, each at least 64 pixels.")
        if p.batch_size < 1 or p.n_iter < 1:
            raise ValueError("Set Batch count and Batch size to at least 1.")

        prompts = p.prompt if isinstance(p.prompt, list) else [p.prompt]
        adapter_names = {
            tag.positional[0]
            for prompt in prompts
            for tag in extra_networks.parse_prompt(prompt or "")[1].get("lora", ())
            if tag.positional
        }
        if adapter_names:
            if p.disable_extra_networks or "lora" not in extra_networks.extra_network_registry:
                raise ValueError("Enable Forge's built-in LoRA extension and allow extra networks to use LoRA or LoKr tags.")
            import networks

            def missing_adapters():
                return adapter_names - networks.available_networks.keys() - {
                    name for name in networks.available_network_aliases
                    if name.lower() not in networks.forbidden_network_aliases
                }

            if missing_adapters():
                networks.list_available_networks()
                missing = missing_adapters()
                if missing:
                    raise ValueError(f"Forge could not find these LoRA or LoKr adapters: {', '.join(sorted(missing))}.")
            failed = adapter_names & networks.failed_networks.keys()
            if failed:
                raise ValueError(f"Forge previously failed to load these adapters: {', '.join(sorted(failed))}. Refresh the Lora list after correcting the files.")

        modules = p.override_settings.get("forge_additional_modules", shared.opts.forge_additional_modules)
        if not isinstance(modules, (list, tuple)) or len(modules) != 2 or not all(isinstance(module, str) for module in modules):
            raise ValueError("Select exactly two modules, the MOSS Qwen3.5 9B encoder and FLUX.2 VAE.")
        selected = {Path(module).name for module in modules}
        if len(selected) != 2 or not selected <= main_entry.module_list.keys() or "sd_vae" in p.override_settings:
            raise ValueError("Select only the MOSS Qwen3.5 9B encoder and FLUX.2 VAE.")
        components = {fingerprint(main_entry.module_list[name]) for name in selected}
        encoders = components & ENCODERS.keys()
        if VAE not in components or not encoders:
            raise ValueError("Select the matching MOSS Qwen3.5 9B encoder and FLUX.2 VAE listed in the README.")

        p.sampler_name = "ER SDE"
        p.scheduler = "Simple"
        p.steps = 25
        p.cfg_scale = 3.0
        p.distilled_cfg_scale = 5.0
        p.s_min_uncond = 0.0
        if p.seed in (-1, "-1", None):
            p.seed = 855373723910865
        p.subseed_strength = 0.0
        p.seed_resize_from_w = p.seed_resize_from_h = -1
        p.negative_prompt = ""
        p.styles = []
        p.enable_hr = False
        p.refiner_checkpoint = None
        p._pi_refiner = None
        p.extra_generation_params.update({
            "Script": TITLE,
            "MOSS Realism Encoder": ENCODERS[next(iter(encoders))],
        })
        p._moss_realism = True
        recipe_options = OPTIONS.copy()
        if "sd_lora" in shared.opts.data_labels:
            recipe_options["sd_lora"] = "None"
        missing_options = recipe_options.keys() - shared.opts.data.keys()
        p.override_settings.update(recipe_options)
        p.override_settings_restore_afterwards = True
        try:
            return processing.process_images(p)
        finally:
            for name in missing_options:
                shared.opts.data.pop(name, None)


class MossRealismSampler(scripts.Script):
    def title(self):
        return "MOSS Base Realism sampler settings"

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def process_before_every_sampling(self, p, *args, **kwargs):
        if getattr(p, "_moss_realism", False) and p.sampler.func is sampling.sample_er_sde:
            p.sampler.func = partial(sampling.sample_er_sde, s_noise=1.1, max_stage=3)
            p.extra_generation_params.update({"Sigma noise": 1.1, "ER SDE max stage": 3})
