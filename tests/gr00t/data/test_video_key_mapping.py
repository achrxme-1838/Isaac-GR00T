"""Camera subsets must read the selected dataset view under the policy's key."""

from dataclasses import asdict
import json
from pathlib import Path
import runpy
from unittest.mock import Mock

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig
import numpy as np
import pytest


@pytest.fixture
def dataset_path(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    cameras = ["left_view", "right_view", "wrist_view"]
    info = {
        "features": {f"observation.images.{key}": {} for key in cameras},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
    }
    modality = {"video": {key: {"original_key": f"observation.images.{key}"} for key in cameras}}
    for name, data in (("info", info), ("modality", modality), ("stats", {})):
        (meta / f"{name}.json").write_text(json.dumps(data))
    (meta / "episodes.jsonl").write_text('{"episode_index": 0, "length": 10}\n')
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "open drawer"}\n')
    return tmp_path


def test_open_drawer_wrist_only_reads_wrist_video(dataset_path, monkeypatch):
    tag = "robocasa_panda_omron"
    # Restore the registry after importing the custom config's override.
    monkeypatch.setitem(MODALITY_CONFIGS, tag, MODALITY_CONFIGS[tag])
    config_path = Path(__file__).resolve().parents[3] / "scripts/open_drawer_wrist_only_config.py"
    config = runpy.run_path(str(config_path))["config"]
    # Check the mapping also survives the JSON round trip used by saved configs.
    config["video"] = ModalityConfig(**json.loads(json.dumps(asdict(config["video"]))))
    loader = LeRobotEpisodeLoader(dataset_path, config)
    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    decode = Mock(return_value=frames)
    monkeypatch.setattr("gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices", decode)

    videos = loader._load_video_data(0, [0])

    assert list(videos) == ["res256_image_wrist_0"]
    assert videos["res256_image_wrist_0"] is frames
    decode.assert_called_once_with(
        str(dataset_path / "videos/chunk-000/observation.images.wrist_view/episode_000000.mp4"),
        [0],
        decoder_kwargs={},
    )


@pytest.mark.parametrize(
    "keys, mapping, expected",
    [
        (["wrist_view"], None, {}),
        (
            ["res256_image_side_0", "res256_image_side_1", "res256_image_wrist_0"],
            None,
            {
                "res256_image_side_0": "left_view",
                "res256_image_side_1": "right_view",
                "res256_image_wrist_0": "wrist_view",
            },
        ),
        (
            ["left_view", "res256_image_wrist_0"],
            {"res256_image_wrist_0": "wrist_view"},
            {"res256_image_wrist_0": "wrist_view"},
        ),
    ],
)
def test_video_key_resolution(dataset_path, keys, mapping, expected):
    video = ModalityConfig(delta_indices=[0], modality_keys=keys, video_key_mapping=mapping)
    loader = LeRobotEpisodeLoader(dataset_path, {"video": video})
    assert loader._video_key_mapping == expected


@pytest.mark.parametrize(
    "mapping, error",
    [
        ({"unknown_camera": "wrist_view"}, "not in configured keys"),
        ({"res256_image_wrist_0": "missing_view"}, "not in dataset keys"),
        ({}, "add them to video_key_mapping"),
    ],
)
def test_invalid_explicit_mapping_fails_early(dataset_path, mapping, error):
    video = ModalityConfig(
        delta_indices=[0],
        modality_keys=["res256_image_wrist_0"],
        video_key_mapping=mapping,
    )
    with pytest.raises(ValueError, match=error):
        LeRobotEpisodeLoader(dataset_path, {"video": video})


def test_unmapped_subset_does_not_guess_camera(dataset_path):
    video = ModalityConfig(delta_indices=[0], modality_keys=["res256_image_wrist_0"])
    with pytest.raises(AssertionError, match="set video_key_mapping explicitly"):
        LeRobotEpisodeLoader(dataset_path, {"video": video})
