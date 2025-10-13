from aurora import AuroraPretrained

def build_model_eu(surf_vars, static_vars, atmos_vars,
                   patch_size=4, bf16=True, stabilise=False, strict=False):
    model = AuroraPretrained(
        surf_vars=tuple(surf_vars),
        static_vars=tuple(static_vars),
        atmos_vars=tuple(atmos_vars),
        patch_size=patch_size,
        bf16_mode=bf16,                  # per docs: experimental, helps memory
        stabilise_level_agg=stabilise,   # per docs: use if grads explode
    )
    model.load_checkpoint(strict=False)  # strict=False if you extend variables
    model.configure_activation_checkpointing()  # per docs
    return model
