"""Serve the RoboCasa team's N1.5 policy to the current rollout client.

Run this standalone script with the official N1.5 fork's Python environment.
It deliberately imports no modules from this N1.7 checkout. The legacy policy
keeps its own transforms and normalization; only the wire interface is adapted.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import traceback

import msgpack
import msgpack_numpy
import numpy as np
import zmq


def _encode(value):
    if isinstance(value, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _decode(value):
    if value.get("__ndarray_class__"):
        return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
    if value.get(b"nd", value.get("nd")) and value.get(b"kind", value.get("kind")) in (
        "O",
        b"O",
    ):
        raise ValueError("Object arrays are not supported")
    return msgpack_numpy.decode(value)


def pack(value):
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack(value):
    return msgpack.unpackb(value, object_hook=_decode, raw=False)


class N15PolicyBridge:
    def __init__(self, policy):
        self.policy = policy
        self.modalities = policy.get_modality_config()

    def dispatch(self, request):
        endpoint = request.get("endpoint")
        if endpoint == "ping":
            return {"status": "ok", "policy": "robocasa365-n15"}
        if endpoint == "get_modality_config":
            # N1.5 keys include video./state./action.; the current contract
            # uses bare names for those modalities and full annotation keys.
            return {
                name: {
                    "__ModalityConfig__": True,
                    "as_json": {
                        "delta_indices": [int(i) for i in config.delta_indices],
                        "modality_keys": [
                            key if name == "language" else key.removeprefix(f"{name}.")
                            for key in config.modality_keys
                        ],
                    },
                }
                for name, config in self.modalities.items()
            }
        if endpoint == "reset":
            # Official N1.5 Gr00tPolicy is stateless; no reset endpoint exists.
            return {}
        if endpoint != "get_action":
            raise ValueError(f"Unknown endpoint: {endpoint}")
        observation = request["data"]["observation"]
        inputs = {}
        for name, config in self.modalities.items():
            if name == "action":
                continue
            for key in config.modality_keys:
                value = observation[key]
                if name == "language":
                    # N1.5 expects (batch, time); current client sends [str].
                    value = np.asarray(value)
                    if value.ndim == 1:
                        value = value[:, None]
                inputs[key] = value
        actions = self.policy.get_action(inputs)
        # Preserve the official denormalized commands, including OSC deltas.
        # No relative-to-absolute transform or additional scaling belongs here.
        return actions, {}


def serve(policy, host: str, port: int):
    bridge = N15PolicyBridge(policy)
    with zmq.Context() as context, context.socket(zmq.REP) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://{host}:{port}")
        print(f"RoboCasa365 N1.5 server ready at tcp://{host}:{port}", flush=True)
        while True:
            message = socket.recv()
            try:
                response = bridge.dispatch(unpack(message))
                reply = pack(response)
            except Exception as exc:
                traceback.print_exc()
                reply = pack({"error": str(exc)})
            socket.send(reply)


def load_policy(model_path: Path, gr00t_n15_path: Path):
    checkout = gr00t_n15_path.expanduser().resolve()
    if not (checkout / "gr00t/model/policy.py").is_file():
        raise FileNotFoundError(
            f"Expected https://github.com/robocasa-benchmark/Isaac-GR00T at {checkout}"
        )
    model_path = model_path.expanduser().resolve()
    model_config = json.loads((model_path / "config.json").read_text())
    if model_config.get("model_type") != "gr00t_n1_5":
        raise ValueError("--model-path must point to a GR00T N1.5 checkpoint directory")
    metadata = json.loads((model_path / "experiment_cfg/metadata.json").read_text())
    if "new_embodiment" not in metadata:
        raise ValueError("Expected RoboCasa365 new_embodiment metadata in checkpoint")
    sys.path.insert(0, str(checkout))
    import gr00t

    if not Path(gr00t.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError(
            "A different gr00t is imported; start this script in a fresh N1.5 process"
        )
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP["panda_omron"]
    return Gr00tPolicy(
        model_path=str(model_path),
        embodiment_tag="new_embodiment",
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=4,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gr00t-n15-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()
    policy = load_policy(args.model_path, args.gr00t_n15_path)
    try:
        serve(policy, args.host, args.port)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
