r"""PandaOmron 링크 길이를 바꿔서 기존 GR00T policy를 GUI로 평가한다.

기존 모델 서버(127.0.0.1:5555)를 켜둔 상태에서 별도 터미널에 복붙:

cd /home/lim/rl_ws/Isaac-GR00T
unset PYTHONPATH PYTHONHOME PYOPENGL_PLATFORM
MUJOCO_GL=glfw \
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/eval_robocasa_gui_x_emb.py \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n-episodes 10 --max-episode-steps 720 --n-action-steps 8 --seed 0 \
  --x_emb '{"link2": 1.2, "link4": 1.2}'

--x_emb '{"link2": 2}' : 상완 길이만 2배
--x_emb '{}'           : 원래 로봇 (옵션 생략 또는 비율 1도 동일)

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
실제 관측값과 영상은 변형된 로봇에서 얻는다. 매 reset에 원본 XML부터 적용하며,
설치된 로봇 파일은 수정하지 않는다. 큰 비율은 초기 충돌이나 제어 불안정을 만들 수 있다.
"""

import argparse
from copy import deepcopy
from functools import partial
import json
import math
import xml.etree.ElementTree as ET

import numpy as np


LINK_AXES = {"link0": 2, "link2": 1, "link3": 0, "link4": 1, "link6": 0, "link7": 2}


def parse_x_emb(value: str) -> dict[str, float]:
    try:
        scales = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(scales, dict):
        raise argparse.ArgumentTypeError("--x_emb must be a JSON object, e.g. '{\"link2\": 1.2}'.")
    for link, ratio in scales.items():
        if link not in LINK_AXES:
            raise argparse.ArgumentTypeError(
                f"Unsupported link {link!r}; choose from: {', '.join(LINK_AXES)}."
            )
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or ratio <= 0
        ):
            raise argparse.ArgumentTypeError(f"{link}: ratio must be a finite positive number.")
    return scales


def _vec(value):
    return np.fromstring(value, sep=" ")


def _text(value):
    return " ".join(f"{component:.12g}" for component in value)


def scale_robot_xml(xml: str, scales: dict[str, float]) -> str:
    """Stretch stock Panda links in the XML before MuJoCo builds physics/controllers."""
    scales = {link: ratio for link, ratio in scales.items() if ratio != 1}
    if not scales:
        return xml
    root = ET.fromstring(xml)
    assets = root.find("asset")
    meshes = {mesh.get("name"): mesh for mesh in assets.findall("mesh")}
    for link, ratio in scales.items():
        body = root.find(f".//body[@name='robot0_{link}']")
        if body is None:
            raise ValueError(
                f"robot0_{link} is missing; --x_emb requires the stock PandaOmron robot."
            )
        stretch = np.ones(3)
        stretch[LINK_AXES[link]] = ratio

        # Only direct attachments move; downstream links keep their own lengths.
        for child in body:
            if "pos" in child.attrib:
                child.set("pos", _text(_vec(child.get("pos")) * stretch))

        scaled_meshes = {}
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name not in scaled_meshes:
                mesh = deepcopy(meshes[mesh_name])
                new_name = f"{mesh_name}_x_emb_{link}"
                mesh.set("name", new_name)
                mesh.set("scale", _text(_vec(mesh.get("scale", "1 1 1")) * stretch))
                assets.append(mesh)
                scaled_meshes[mesh_name] = new_name
            geom.set("mesh", scaled_meshes[mesh_name])

        # Stock Panda inertias are diagonal in the body frame. Affine stretch at fixed mass:
        # C = integral(r r^T dm), I = trace(C) * identity - C.
        inertial = body.find("inertial")
        inertia = _vec(inertial.get("diaginertia"))
        second_moments = (inertia.sum() / 2 - inertia) * stretch**2
        inertial.set("diaginertia", _text(second_moments.sum() - second_moments))
    return ET.tostring(root, encoding="unicode")


def configure_x_emb(env, scales):
    robot_env = env.unwrapped.env
    robot_env.hard_reset = True
    robot_env.set_xml_processor(partial(scale_robot_xml, scales=scales))


def main():
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
    parser.add_argument(
        "--x_emb", "--x-emb", type=parse_x_emb, default={}, help="JSON link length ratios"
    )
    args = parser.parse_args()
    if min(args.n_episodes, args.max_episode_steps, args.n_action_steps) < 1:
        parser.error("Episode and step counts must be positive.")
    if not args.env_name.startswith("robocasa_panda_omron/"):
        parser.error("This script supports robocasa_panda_omron environments only.")

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
    with PolicyClient(
        host=args.policy_client_host, port=args.policy_client_port, timeout_ms=120000
    ) as policy:
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
        gui = GuiWrapper(gym.make(args.env_name, enable_render=True, seed=args.seed))
        try:
            configure_x_emb(gui, args.x_emb)
            env = MultiStepWrapper(
                gui,
                contract=contract,
                max_episode_steps=args.max_episode_steps,
                terminate_on_success=True,
            )
            run_episodes(env, gui, policy, args.n_episodes, args.seed)
        except KeyboardInterrupt:
            print("Viewer stopped.")
        finally:
            gui.close()


if __name__ == "__main__":
    main()
