r"""PandaOmron 링크 길이를 바꿔서 영상과 성공률을 저장한다.

기존 모델 서버(127.0.0.1:5555)를 켜두고 별도 터미널에 복붙:

cd /home/lim/rl_ws/Isaac-GR00T
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PATH="/usr/bin:/bin" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  gr00t/eval/rollout_policy_x_emb.py \
  --policy-client-host 127.0.0.1 --policy-client-port 5555 \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n-episodes 10 --n-envs 1 --max-episode-steps 720 --n-action-steps 8 \
  --video-dir "$PWD/eval_videos/open_drawer_x_emb" --seed 0 \
  --x_emb '{"link2": 1.2, "link4": 1.2}'

영상 없이 성공률만 저장하려면 위 명령 마지막에 --no_video 추가.
결과: --video-dir 폴더의 results_날짜-시간.json (success_rate는 0~1).
실행마다 새 파일에 성공 횟수, 실제 에피소드 수, 에피소드별 결과와 설정을 기록한다.
--x_emb '{}' 또는 비율 1은 원래 로봇. obs/action 형식과 좌표계는 그대로 유지한다.

이름 / 늘어나는 구간 / 로컬 축 / 원래 축 방향 길이:
  link0 : 팔 밑동 -> 관절 1       / Z / 0.333 m
  link2 : 상완, 관절 2 -> 3       / Y / 0.316 m
  link3 : 팔꿈치 옆 오프셋, 3 -> 4 / X / 0.0825 m
  link4 : 전완, 관절 4 -> 5       / Y / 0.384 m
  link6 : 손목 옆 오프셋, 6 -> 7   / X / 0.088 m
  link7 : 관절 7 -> 손 부착점      / Z / 0.1065 m
link1, link5는 다음 관절까지 위치 오프셋이 0이므로 제외한다.
길이 축 밖의 성분은 유지한다. 변경 방식은 eval_robocasa_gui_x_emb.py와 동일하다.
"""

import argparse
from dataclasses import dataclass

from gr00t.eval.rollout_policy import RolloutConfig, main as run_rollout
from gr00t.eval.sim.robocasa_x_emb import parse_x_emb
import tyro


@dataclass
class XEmbRolloutConfig(RolloutConfig):
    env_name: str = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
    n_envs: int = 1
    x_emb: str = "{}"
    """JSON link length ratios, e.g. '{"link2": 1.2, "link4": 1.2}'."""


def main():
    args = tyro.cli(XEmbRolloutConfig)
    try:
        scales = parse_x_emb(args.x_emb)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"--x_emb: {exc}") from exc
    print(f"Link length ratios: {scales or 'original robot'}", flush=True)
    return run_rollout(args, x_emb=scales)


if __name__ == "__main__":
    main()
