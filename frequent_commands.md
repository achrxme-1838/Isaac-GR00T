# 자주 쓰는 RoboCasa 명령어

아래 명령어는 모두 로컬 PC에서 실행한다.
현재 예시는 `open_drawer_n17_286967/checkpoint-60000`을 사용한다.

## 1. 서버에서 체크포인트 다운로드

`~/.ssh/config`에 등록한 `target-server`를 통해 접속한다.
학습 폴더 전체를 복사하며, 중간에 끊기면 같은 명령어를 다시 실행한다.

```bash
mkdir -p /home/lim/rl_ws/Isaac-GR00T/outputs

rsync -avh --partial --progress -e ssh \
  target-server:/home/visitor_km/shlim/Isaac-GR00T/outputs/open_drawer_n17_286967 \
  /home/lim/rl_ws/Isaac-GR00T/outputs/
```

## 2. 모델 서버 실행 — 터미널 1

GUI 평가와 비디오 저장 모두 이 서버가 필요하다.
기존 서버에서 다른 모델을 사용 중이면 해당 터미널에서 `Ctrl+C`로 종료한 뒤 실행한다.
모델 로딩이 끝나면 서버를 켜둔 채 별도 터미널에서 3번 또는 4번을 실행한다.

```bash
conda activate gr00t
cd /home/lim/rl_ws/Isaac-GR00T

python gr00t/eval/run_gr00t_server.py \
  --model-path "$PWD/outputs/open_drawer_n17_286967/checkpoint-60000" \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5555
```

## 3. 시뮬레이터 GUI로 보기 — 터미널 2

MuJoCo 창에서 시점을 조절하며 10개 에피소드를 확인한다.
창을 닫거나 `Ctrl+C`를 누르면 종료된다. 이 명령어는 비디오를 저장하지 않는다.

```bash
cd /home/lim/rl_ws/Isaac-GR00T
unset PYTHONPATH PYTHONHOME PYOPENGL_PLATFORM

MUJOCO_GL=glfw \
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/eval_robocasa_gui.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n-episodes 10 \
  --max-episode-steps 720 \
  --n-action-steps 8 \
  --seed 0
```

## 4. 평가 결과를 비디오로 저장 — 터미널 2

GUI 창 없이 10개 에피소드를 평가하고, 측면 카메라 2개와 손목 카메라 1개를
가로로 합친 MP4를 저장한다. 마지막에 터미널에 성공률이 출력되고,
같은 폴더의 `results_날짜-시간.json`에도 기록된다. 파일마다 성공 횟수,
실제 평가한 에피소드 수, 에피소드별 결과와 실행 설정이 들어간다.
Conda `ffmpeg`의 C++ 라이브러리 충돌을 피하도록 이 평가 프로세스에서는
시스템 `ffmpeg`를 사용하고 외부 라이브러리 경로 설정을 해제한다.

```bash
cd /home/lim/rl_ws/Isaac-GR00T

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PATH="/usr/bin:/bin" \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  gr00t/eval/rollout_policy.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n-episodes 10 \
  --n-envs 1 \
  --max-episode-steps 720 \
  --n-action-steps 8 \
  --video-dir "$PWD/eval_videos/open_drawer_60000" \
  --seed 0
```

저장 위치: `/home/lim/rl_ws/Isaac-GR00T/eval_videos/open_drawer_60000/`

기본 녹화 설정은 2스텝마다 한 프레임을 저장하고 20fps로 재생하므로,
시뮬레이션 시간 대비 약 2배속이다.

영상 없이 성공률만 기록하려면 위 명령에 `--no_video`를 추가한다.
이때도 `--video-dir`로 지정한 폴더에 결과 JSON이 저장된다.

## 5. 링크 길이를 바꿔서 영상·성공률 저장 — 터미널 2

상완(`link2`)과 전완(`link4`)을 각각 1.2배로 늘린다.
GUI 버전과 같은 변형이며 policy obs/action 형식은 유지한다.

```bash
cd /home/lim/rl_ws/Isaac-GR00T

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PATH="/usr/bin:/bin" \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  gr00t/eval/rollout_policy_x_emb.py \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n-episodes 10 --n-envs 1 \
  --max-episode-steps 720 --n-action-steps 8 \
  --video-dir "$PWD/eval_videos/open_drawer_x_emb" \
  --seed 0 \
  --x_emb '{"link2": 1.2, "link4": 1.2}'
```

이 명령에도 `--no_video`를 추가하면 영상 없이 결과 JSON만 저장한다.
`success_rate`는 0~1 사이 값이다. 예를 들어 `0.7`은 성공률 70%이다.

## 다른 모델 또는 환경으로 변경

- 모델 변경: 2번의 `--model-path`를 수정하고 모델 서버를 다시 실행한다.
- 환경 변경: 3번 또는 4번의 `--env-name`을 수정한다.
  예: `robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env`
- 영상 저장 폴더 변경: 4번의 `--video-dir`를 수정한다.
