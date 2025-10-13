# aurora-europe/projects/cerra_hr/models/build_model_eu.py
from typing import Iterable, Optional, Sequence, Tuple
from aurora import AuroraPretrained
import torch

def build_model_eu(
    surf_vars: Sequence[str],
    static_vars: Sequence[str],
    atmos_vars: Sequence[str],           # includes 'rh' instead of 'q'
    *,
    patch_size: int = 4,
    bf16: bool = True,
    stabilise: bool = False,
    strict: bool = False,                # q->rh extension => keep False
    zero_init_rh_encoder: bool = True,   # gently introduce new var
    ckpt_path: Optional[str] = None,     # load your own fine-tuned ckpt if provided
) -> AuroraPretrained:
    """
    Build AuroraPretrained for Europe CERRA path-B fine-tuning.

    - Uses bf16_mode (experimental) + activation checkpointing per official docs.
    - Loads checkpoint with strict=False because we changed the variable set (q -> rh).
    - Optionally zero-inits the encoder token embedding for 'rh' to avoid perturbing existing vars.
    - Optionally loads a specific checkpoint path (e.g., your own fine-tuned weights).
    """
    model = AuroraPretrained(
        surf_vars=tuple(surf_vars),
        static_vars=tuple(static_vars),
        atmos_vars=tuple(atmos_vars),
        patch_size=patch_size,
        bf16_mode=bf16,                   # EXPERIMENTAL per docs
        stabilise_level_agg=stabilise,    # set True if gradients explode
    )

    # Load weights: if a specific ckpt is passed, prefer it; otherwise load default.
    # We keep strict=False for q->rh extension per official guidance.
    if ckpt_path:
        model.load_checkpoint(strict=False, path=ckpt_path)
    else:
        model.load_checkpoint(strict=False)

    # Save memory: activation checkpointing per docs
    model.configure_activation_checkpointing()

    # (Optional) zero-init encoder token for the NEW 'rh' variable to reduce drift
    if zero_init_rh_encoder:
        try:
            if "rh" in atmos_vars and hasattr(model, "encoder"):
                idx = list(atmos_vars).index("rh")
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
