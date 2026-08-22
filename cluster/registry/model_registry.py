"""
External presets so you don't touch the repo's model_registry.py.
Registers:
  - nebel-raven-1b-mig   (1.0B, ctx=2048)  — good starter for 39GB MIG
  - nebel-raven-0p7b-mig (0.7B, ctx=4096)  — safer memory for longer context
"""
def _cfg_1b():
    return dict(
        name="nebel-raven-1b-mig",
        hf_config=dict(org="local", name="nebel-raven-1b-mig"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=4096,
        tie_embeddings=True,
        n_embd=2560,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=10240,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="SandwichBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1.0e-6,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="takase",
        init_orthogonal=False,
        state_init="like-init",
        injection_type="linear",
        n_layers_in_recurrent_block=4,
        mean_recurrence=24,
        sampling_scheme="poisson-lognormal-filling",
        mean_backprop_depth=8,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        qk_bias=True,
        activation_checkpoint_impl="per-iteration",
    )

def _cfg_0p7b():
    return dict(
        name="nebel-raven-0p7b-mig",
        hf_config=dict(org="local", name="nebel-raven-0p7b-mig"),
        block_size=4096,                  # longer context on 39GB MIG
        vocab_size=32000,
        padding_multiple=4096,
        tie_embeddings=True,
        n_embd=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=8192,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="SandwichBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1.0e-6,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="takase",
        init_orthogonal=False,
        state_init="like-init",
        injection_type="linear",
        n_layers_in_recurrent_block=4,
        mean_recurrence=24,
        sampling_scheme="poisson-lognormal-filling",
        mean_backprop_depth=8,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        qk_bias=True,
        activation_checkpoint_impl="per-iteration",
    )

def install():
    import recpre.model_registry as mr
    add = [_cfg_1b(), _cfg_0p7b()]
    def _put(cfg):
        if hasattr(mr, "registry"):
            if isinstance(mr.registry, list):
                mr.registry[:] = [e for e in mr.registry
                                  if not (isinstance(e, dict) and e.get("name")==cfg["name"])]
                mr.registry.append(cfg)
            elif isinstance(mr.registry, dict):
                mr.registry[cfg["name"]] = cfg
            else:
                mr.registry = [cfg]
        elif hasattr(mr, "REGISTRY"):
            mr.REGISTRY[cfg["name"]] = cfg
        elif hasattr(mr, "register"):
            mr.register(cfg)
        else:
            mr.registry = [cfg]
    for c in add: _put(c)
