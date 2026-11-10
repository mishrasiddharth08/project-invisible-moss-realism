from functools import lru_cache, partial
from pathlib import Path

from k_diffusion import sampling
from modules import hashes, processing, scripts, sd_models, sd_samplers, shared
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
}


@lru_cache(maxsize=16)
def _hash_file(path, size, modified_ns, changed_ns):
    return hashes.calculate_sha256_real(path)


def fingerprint(filename):
    path = Path(filename).resolve(strict=True)
    stat = path.stat()
    return _hash_file(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class MossRealism(scripts.Script):
    def title(self):
        return TITLE

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        return []

    def run(self, p, *args):
        if "ER SDE" not in sd_samplers.all_samplers_map or not OPTIONS.keys() <= shared.opts.data_labels.keys():
            raise RuntimeError("MOSS Base Realism requires Forge Neo with native ER SDE and skip-layer guidance.")

        checkpoint_name = p.override_settings.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint)
        checkpoint = sd_models.checkpoint_aliases.get(checkpoint_name)
        if checkpoint is None or fingerprint(checkpoint.filename) != BASE:
            raise ValueError("Choose the Comfy-Org MOSS-Image Base INT8 ConvRot checkpoint.")

        modules = p.override_settings.get("forge_additional_modules", shared.opts.forge_additional_modules)
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
        p.width, p.height = 1184, 1776
        p.batch_size = p.n_iter = 1
        if p.seed in (-1, "-1", None):
            p.seed = 855373723910865
        p.subseed_strength = 0.0
        p.seed_resize_from_w = p.seed_resize_from_h = -1
        p.negative_prompt = ""
        p.styles = []
        p.disable_extra_networks = True
        p.enable_hr = False
        p.refiner_checkpoint = None
        p.extra_generation_params.update({
            "Script": TITLE,
            "MOSS Realism Encoder": ENCODERS[next(iter(encoders))],
        })
        p._moss_realism = True
        missing_options = OPTIONS.keys() - shared.opts.data.keys()
        p.override_settings.update(OPTIONS)
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
