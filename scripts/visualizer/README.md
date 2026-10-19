# Dynamics model replay

`viz_robocasa_dynamics_model.py` 상단의 `CONFIG = VizConfig(...)`를 수정한 후 실행합니다.

```bash
cd /home/lim/rl_ws/Isaac-GR00T
env -u PYTHONPATH -u PYTHONHOME -u PYOPENGL_PLATFORM MUJOCO_GL=egl \
  gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/visualizer/viz_robocasa_dynamics_model.py
```

기존 RoboCasa 환경과 `mobile_crossemb` checkout이 필요합니다. 현재 머신에는 추가
의존성도 설치했습니다. 새 환경에서는 NumPy 1.26과 호환되는 PyArrow를 사용합니다:

```bash
uv pip install --python gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  'pyarrow==19.0.1' 'wandb>=0.17'
```

`wandb`는 기존 checkpoint loader의 import 의존성입니다. 이 스크립트는 W&B를 초기화하거나
로그인하지 않으며 policy 서버도 사용하지 않습니다.

- 실제 로봇: 기록된 관절·그리퍼 qpos와 base pose를 매 frame 직접 설정합니다.
  `env.step`/`mj_step` 없이 `mj_forward`로 pose만 갱신합니다.
- 초록색 반투명 EEF: 초기 history 이후에는 예측 state를 재귀적으로 입력하고,
  action만 데이터에서 가져옵니다. `mobile_crossemb/source/evaluation`의 rollout을 공유합니다.
  팔 전체가 아닌 hand/gripper mesh만 표시하며, 충돌·물리 모델에는 추가하지 않습니다.
- EEF position은 grip site, quaternion은 `right_hand` body의 xyzw입니다.
  표시할 때만 quaternion을 정규화하며 rollout 출력에는 적용하지 않습니다.
  손가락 개폐는 실제 로봇을 따릅니다. 관절만 예측하는 모델은 FK로 EEF를 구합니다.
- 입력 state 보완과 world-frame 변환에 dataset 값이 필요하면 매 step 경고를 출력하고
  `feedback_warnings.jsonl`에도 기록합니다. `_b` 예측을 `_w`로 표시할 때에는 예측에
  base pose가 없으면 해당 frame의 기록된 base pose를 사용합니다.
- 현재 데이터에는 scene XML, 서랍·물체의 qpos가 없습니다. 주방은 `scene_seed`로 만든
  정적인 참고 배경이며 원본 demonstration의 물체 움직임을 재현하지 않습니다.
  로봇 EEF가 기록된 world pose와 일치하는지는 매 frame 검증합니다.

주요 Python 설정:

| 설정 | 의미 |
| --- | --- |
| `checkpoint_path` | 절대 경로 또는 `dynamics_repo` 기준 checkpoint 경로 |
| `data_path` | `None`이면 checkpoint에 저장된 데이터 경로 |
| `num_trajectories` / `rollout_steps` | 서로 다른 trajectory 수 / 재귀 예측 횟수 |
| `split` / `seed` | 기본 `val`; checkpoint의 split을 사용하며 sampling 재현 가능 |
| `repeats` | 샘플링한 trajectory 묶음의 반복 재생 횟수 |
| `playback_speed` / `pause_seconds` | 기본 실시간 1배속 / GUI 시작·끝 frame 대기 시간 |
| `record_video` | `False`: GUI, `True`: GUI 없이 trajectory별 MP4 저장 |
| `record_camera` | 기본 `robot0_agentview_left`; GUI에서도 같은 카메라로 시작 |
| `ghost_rgba` | 초록색과 투명도 |

초기 history 길이와 state 선택은 checkpoint에서 가져옵니다. 처음에는 실제 EEF와
예측 EEF가 겹치고 이후 prediction을 따라 분리됩니다. `all` 모델은 첫 action의 예측을
매 frame 재귀 적용합니다. `last` + multi-action 모델은 action horizon만큼 frame을
건너뛰어 표시하고 재생 시간에도 반영합니다. 이 모드에 history까지 2 이상이면 기존
평가기와 동일하게 지원하지 않는 조합임을 알립니다.

결과는 `outputs/dynamics_visualizer/<run>/`에 저장합니다:
`config.json`(sampling/checkpoint), `trajectory_*.npz`(기록 state/예측 EEF/원본 model 출력),
`feedback_warnings.jsonl`, `replay_validation.json`(실제 로봇 pose 복원 오차),
그리고 `record_video=True`일 때 `repeat_*_trajectory_*.mp4`.
GUI 창을 닫거나 Ctrl+C로 중단할 수 있습니다.
