"""Minimal load-and-infer test: verify pi05-DROID fits in 8GB VRAM."""
import time
import numpy as np
import os

# Prevent JAX from grabbing all VRAM upfront
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies import droid_policy
from openpi.shared import download
import jax

print("JAX:", jax.__version__, "devices:", jax.devices())

t = time.time()
cfg = _config.get_config("pi05_droid")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
print(f"ckpt dir: {ckpt}")
policy = _policy_config.create_trained_policy(cfg, ckpt)
print(f"policy loaded in {time.time()-t:.1f}s")

# Dummy infer
ex = droid_policy.make_droid_example()
ex["prompt"] = "pick up the red block"
t = time.time()
out = policy.infer(ex)
print(f"infer #1: {time.time()-t:.2f}s  actions={out['actions'].shape}")

t = time.time()
out = policy.infer(ex)
print(f"infer #2: {time.time()-t:.2f}s  actions={out['actions'].shape}")

print("actions[0]:", out["actions"][0])
