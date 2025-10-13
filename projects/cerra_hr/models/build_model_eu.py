from aurora import AuroraPretrained
import torch

def build_model_eu(
    surf_vars,
    static_vars,
    atmos_vars,
    patch_size=4,
    bf16=True,
    stabilise=False,
    strict=False,                 # we extended vars (q->rh), so keep False in calls
    zero_init_rh_encoder=False,   # <-- NEW: optionally zero-init 'rh' encoder token
):
    model = AuroraPretrained(
        surf_vars=tuple(surf_vars),
        static_vars=tuple(static_vars),
        atmos_vars=tuple(atmos_vars),
        patch_size=patch_size,
        bf16_mode=bf16,
        stabilise_level_agg=stabilise,
    )
    # Extended var set -> strict=False per docs
    model.load_checkpoint(strict=False)
    model.configure_activation_checkpointing()

    if zero_init_rh_encoder:
        try:
            var_list = list(atmos_vars)
            if "rh" in var_list and hasattr(model, "encoder"):
                idx = var_list.index("rh")
                token_embeds = getattr(model.encoder, "atmos_token_embeds", None)
                if token_embeds is not None and hasattr(token_embeds, "weights"):
                    with torch.no_grad():
                        token_embeds.weights[idx].zero_()
                    print("Zero-initialized encoder token embedding for 'rh'.")
                else:
                    print("Warning: encoder.atmos_token_embeds.weights not found; skip zero-init.")
            else:
                print("Warning: 'rh' not in atmos_vars or encoder missing; skip zero-init.")
        except Exception as e:
            print(f"Warning: could not zero-init 'rh' token embedding: {e}")

    return model
