r"""PandaOmron 링크 길이를 바꿔서 기존 GR00T policy를 GUI로 평가한다.

기존 모델 서버(127.0.0.1:5555)를 켜둔 상태에서 별도 터미널에 복붙:

cd /home/lim/rl_ws/Isaac-GR00T
unset PYTHONPATH PYTHONHOME PYOPENGL_PLATFORM
MUJOCO_GL=egl \
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/eval_robocasa_gui_x_emb.py \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n-episodes 10 --max-episode-steps 720 --n-action-steps 8 --seed 0 \
  --x_emb '{"link2": 1.2, "link4": 1.2}' \
  --use_exp_adap=True --record_vid

--x_emb '{"link2": 2}' : 상완 길이만 2배
--x_emb '{}'           : 원래 로봇 (옵션 생략 또는 비율 1도 동일)
--use_exp_adap=True    : 변형된 팔 FK -> 원래 팔의 위치·자세 IK로 관절 관측 변환
                        원래 길이의 팔 + 가상 관절각을 반투명 청록색으로 겹쳐 표시
--use_ext_adap         : --use_exp_adap=True와 동일한 별칭
--use_exp_adap=False   : 변형된 팔의 실제 관측 사용 (기본값)
--record_vid          : GUI 없이 왼쪽 측면 뷰(팔 + 작업 대상)를 960x720 MP4로 저장
                        eval_videos/gui_<시각>/episode_NNN.mp4 (반투명 팔도 포함)
--video-dir PATH      : 영상 저장 폴더 지정
--record-camera robot0_agentview_right : 오른쪽 측면 뷰로 녹화

지원하는 이름 / 늘어나는 구간 / 링크 로컬 축 / 원래 축 방향 길이:
  link0 : 팔 밑동 -> 관절 1       / Z / 0.333 m
  link2 : 상완, 관절 2 -> 관절 3  / Y / 0.316 m
  link3 : 팔꿈치 옆 오프셋, 3 -> 4 / X / 0.0825 m
  link4 : 전완, 관절 4 -> 관절 5  / Y / 0.384 m
  link6 : 손목 옆 오프셋, 6 -> 7   / X / 0.088 m
  link7 : 관절 7 -> 손 부착점      / Z / 0.1065 m
link1, link5는 다음 관절까지 위치 오프셋이 0이므로 길이 변경 대상에서 제외한다.
이 이름들은 MJCF의 robot0_linkN body에 해당한다. 축은 월드 좌표계가 아니다.
link4의 X 오프셋(-0.0825 m)처럼 길이 축 밖의 성분은 그대로 둔다.

관절 간격, 시각/충돌 메시, 질량중심과 관성만 변경한다. 질량은 유지한다.
policy obs/action 키·차원·좌표계·스케일 및 controller 설정은 기존과 동일하다.
use_exp_adap=True이면 joint_position, joint_position_cos/sin을 원래 팔의 IK 해로
교체하고 joint_velocity도 Jacobian으로 환산한다. IK는 팔 밑동(link0) 좌표계에서
실제 grip_site의 위치와 자세를 맞춘다. 이전 IK 해에서 시작하며 관절 한계를 지킨다.
원래 팔의 도달 범위를 벗어나거나 IK가 수렴하지 않으면 근사해를 사용하고 경고한다.
에피소드 종료 시 IK 최대 위치/회전 오차와 허용치 초과 횟수·비율을 출력한다.
집계는 reset 관측 1회와 모든 내부 환경 스텝을 포함한다(720스텝이면 721회).
카메라 영상, 말단·베이스 관측, gripper 관측은 실제 값을 유지한다. 액션은 변경하지 않는다.
반투명 팔은 GUI와 녹화 영상에 표시되며 정책 카메라 영상과 충돌·물리 계산에는 포함되지 않는다.
--record_vid 사용 시 GUI 창을 열지 않는다. 생략하면 GUI만 표시하며 영상은 저장하지 않는다.
매 스텝 정책에 전달한 관절각을 표시하고 베이스 위치와 그리퍼 개폐는 실제 로봇을 따른다.
관측 변환은 물리 모델을 바꾸지 않는다. 매 reset에 원본 XML부터 링크 비율을 적용하며,
설치된 로봇 파일은 수정하지 않는다. 큰 비율은 초기 충돌이나 제어 불안정을 만들 수 있다.
"""

import argparse

from gr00t.eval.sim.gui_video_recorder import add_recording_args, recording_directory
from gr00t.eval.sim.robocasa_x_emb import configure_x_emb, parse_x_emb


def parse_bool(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected True or False.")
    return value.lower() == "true"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-name", default="robocasa_panda_omron/OpenDrawer_PandaOmron_Env")
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    add_recording_args(parser)
    parser.add_argument(
        "--x_emb", "--x-emb", type=parse_x_emb, default={}, help="JSON link length ratios"
    )
    parser.add_argument(
        "--use_exp_adap",
        "--use-exp-adap",
        "--use_ext_adap",
        "--use-ext-adap",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Adapt joint observations and overlay a translucent nominal arm in the viewer.",
    )
    args = parser.parse_args(argv)
    if min(args.n_episodes, args.max_episode_steps, args.n_action_steps) < 1:
        parser.error("Episode and step counts must be positive.")
    if not args.env_name.startswith("robocasa_panda_omron/"):
        parser.error("This script supports robocasa_panda_omron environments only.")
    return args


def main():
    args = parse_args()

    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from gr00t.policy.server_client import PolicyClient
    import gymnasium as gym
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    if __package__:
        from .eval_robocasa_gui import GuiWrapper, run_episodes
    else:
        from eval_robocasa_gui import GuiWrapper, run_episodes

    print(f"Link length ratios: {args.x_emb or 'original robot'}", flush=True)
    print(f"Expert observation adaptation: {args.use_exp_adap}", flush=True)
    with PolicyClient(
        host=args.policy_client_host, port=args.policy_client_port, timeout_ms=120000
    ) as policy:
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
        raw_env = gym.make(args.env_name, enable_render=True, seed=args.seed)
        configure_x_emb(raw_env, args.x_emb)
        overlay_factory = None
        if args.use_exp_adap:
            from gr00t.eval.sim.robocasa_exp_adaptation import ExpertAdaptationWrapper
            from gr00t.eval.sim.robocasa_ghost import NominalArmGhost

            raw_env = ExpertAdaptationWrapper(raw_env, args.x_emb)
            overlay_factory = NominalArmGhost
        gui = GuiWrapper(
            raw_env,
            overlay_factory=overlay_factory,
            video_dir=recording_directory(args),
            record_camera=args.record_camera,
        )
        try:
            env = MultiStepWrapper(
                gui,
                contract=contract,
                max_episode_steps=args.max_episode_steps,
                terminate_on_success=True,
            )
            run_episodes(env, gui, policy, args.n_episodes, args.seed)
        except KeyboardInterrupt:
            print("Evaluation stopped.")
        finally:
            gui.close()


if __name__ == "__main__":
    main()
