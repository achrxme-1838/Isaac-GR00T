# RoboCasa365 + the RoboCasa team's GR00T N1.5

The rollout entry point supports `--mode robocasa365-n15`. It uses the official
RoboCasa365 `RoboCasaGymEnv` with PandaOmron, three native 256×256 RGB cameras,
the `target` scene/object split, and the checkpoint's state/action conventions.
The default task is `OpenCabinet`, execution stride is 16 actions, and episode
length comes from RoboCasa365's task horizon registry. Explicit CLI values
override these defaults.

The simulator viewer and a companion language/three-camera window open by
default. `--instruction "..."` overrides the language sent to the policy.
`--no-visualize` runs headless; policy cameras are still rendered. Recording is
disabled unless `--video-dir` is supplied. G1 EEF/ghost overlays and WBC robot
settings apply only to the default `--mode real-g1`.

## One-time setup

Run these commands from this Isaac-GR00T checkout. The N1.5 model dependencies
and RoboCasa365 simulator dependencies use separate environments.

First set up the simulator using the existing repository installer (includes
kitchen assets):

```bash
INSTALL_FLASH_ATTN=0 bash gr00t/eval/sim/robocasa365/setup_RoboCasa365.sh
```

The simulator Python is
`gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python`. For the companion
window it also needs working `tkinter`/Tk. Use `--no-visualize` on a headless
machine. The installer recreates its dedicated `robocasa365_uv` environment;
run it once, not before every rollout.

Set up the [RoboCasa team's N1.5 fork](https://github.com/robocasa-benchmark/Isaac-GR00T)
in a sibling checkout. These commands use its own dependencies, including
Transformers 4.51.3, without changing the current N1.7 environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/robocasa-benchmark/Isaac-GR00T ../Isaac-GR00T-RoboCasa-N15
git -C ../Isaac-GR00T-RoboCasa-N15 checkout 9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10
uv venv ../Isaac-GR00T-RoboCasa-N15/.venv --python 3.10
uv pip install --python ../Isaac-GR00T-RoboCasa-N15/.venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 setuptools wheel packaging
uv pip install --python ../Isaac-GR00T-RoboCasa-N15/.venv/bin/python \
  -e ../Isaac-GR00T-RoboCasa-N15 diffusers==0.30.2 \
  msgpack==1.1.0 msgpack-numpy==0.4.8
uv pip install --python ../Isaac-GR00T-RoboCasa-N15/.venv/bin/python \
  --no-build-isolation flash-attn==2.7.1.post4
```

The FlashAttention install requires a compatible CUDA toolchain or wheel.
`GIT_LFS_SKIP_SMUDGE=1` skips the fork's optional demo media; its upstream LFS
server no longer serves at least one of these files. Policy weights are
downloaded separately below.

### Repair dependency installation in an existing Python 3.12 environment

If the N1.5 `.venv` already uses Python 3.12, the fork's `onnx==1.15.0`
dependency has no matching Linux x86_64 wheel. Installation attempts a source
build, which can fail when Conda's `protoc` loads an incompatible system
`libstdc++` (`CXXABI_1.3.15` missing). Other dependencies such as Transformers
may remain uninstalled when that build fails.

Use the [ONNX 1.16.2 Python 3.12 wheel](https://pypi.org/project/onnx/1.16.2/#files)
with an explicit uv dependency override. Run this in Bash from the N1.7
checkout, using the existing N1.5 environment with PyTorch and FlashAttention
already installed:

```bash
N15_DIR="$(realpath ../Isaac-GR00T-RoboCasa-N15)"
N15_PY="$N15_DIR/.venv/bin/python"

uv pip install --python "$N15_PY" \
  --overrides <(printf 'onnx==1.16.2\n') \
  --only-binary onnx \
  -e "$N15_DIR" \
  torch==2.5.1 torchvision==0.20.1 \
  numpy==1.26.4 transformers==4.51.3 diffusers==0.30.2 \
  huggingface-hub==0.34.4 \
  msgpack==1.1.0 msgpack-numpy==0.4.8 \
  "setuptools<81" wheel packaging

"$N15_PY" - <<'PY'
import torch
import transformers
import flash_attn
import onnx

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("FlashAttention:", flash_attn.__version__)
print("ONNX:", onnx.__version__)
assert torch.cuda.is_available(), "CUDA GPU is unavailable"
print("GPU:", torch.cuda.get_device_name(0))
PY
```

`--only-binary onnx` prevents another ONNX source build. The override replaces
the fork's declared ONNX pin for this installation only; include it when
reinstalling the fork in this Python 3.12 environment. Dependency resolution
with this override and the binary-only requirement has been checked; a full
installation and model rollout still need to complete on the target machine.

## Download a finetuned checkpoint

The [official benchmark](https://robocasa.ai/docs/build/html/benchmarking/foundation_model_learning.html)
publishes separate checkpoints for `atomic_seen`, `composite_seen`, and
`composite_unseen`. `OpenCabinet` uses the `atomic_seen` checkpoint below.
Download only inference files, without the optimizer/training state:

```bash
CKPT_SUBDIR=gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000
hf download robocasa/robocasa365_checkpoints \
  --revision c484448aba1a9b60a04c9b0ca117241518ea69f3 \
  --include "$CKPT_SUBDIR/config.json" "$CKPT_SUBDIR/experiment_cfg/*" \
    "$CKPT_SUBDIR/model*.safetensors*" \
  --local-dir checkpoints/robocasa365
```

## Run

Terminal 1, from this checkout, using the N1.5 interpreter:

```bash
../Isaac-GR00T-RoboCasa-N15/.venv/bin/python \
  gr00t/experiment/custom/serve_robocasa365_n15.py \
  --gr00t-n15-path ../Isaac-GR00T-RoboCasa-N15 \
  --model-path checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000 \
  --port 5556
```

Wait for `RoboCasa365 N1.5 server ready`, then run Terminal 2:

```bash
gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python \
  gr00t/experiment/custom/run_gr00t_on_robocas.py \
  --mode robocasa365-n15 \
  --task OpenCabinet \
  --policy-host 127.0.0.1 --policy-port 5556 \
  --robocasa-split target \
  --n-episodes 1 --n-action-steps 16
```

For a simulator-only check, omit the server and add
`--smoke-test --max-episode-steps 20`. This sends zero motion deltas with the
gripper open; it does not evaluate the policy.

Use `--robocasa365-path /path/to/robocasa` for another installed upstream
checkout. Do not point it at the GR1 tabletop fork. Choose a task covered by
the downloaded checkpoint; changing `--task` does not switch model weights.

## Compatibility

`serve_robocasa365_n15.py` loads `panda_omron` and `new_embodiment` from the
official N1.5 fork. Its standalone bridge adapts N1.5's modality names,
language batch/time dimensions, and action response to the current client's
MessagePack interface. The model's own transforms handle normalization and
denormalization. OSC translation/rotation deltas and base commands go through
the official simulator wrapper without another coordinate transform.

The original N1.5 `scripts/run_eval.py --server` uses a different Torch-based
wire protocol and cannot replace this bridge. The N1.7
`gr00t/eval/run_gr00t_server.py` cannot load this N1.5 checkpoint. Port 5556 is
the kitchen mode's default, allowing the existing G1 server on 5555 to remain
available.

CPU coverage includes actual ZMQ exchange with a fake N1.5 policy, language and
camera forwarding, schema rejection, zero-delta smoke actions, partial action
chunks, and cleanup. Task success still needs a rollout with installed assets
and the downloaded model.
