# 다른 Ubuntu GPU 서버에 GR00T 설치하기

대상은 **Ubuntu 22.04/24.04 + NVIDIA GPU 일반 서버**다. 기존
[`Dockerfile`](Dockerfile)의 CUDA 12.8 / Python 3.12 / PyTorch 의존성을 사용한다.
호스트에는 NVIDIA 드라이버, Docker Engine, Compose 플러그인,
NVIDIA Container Toolkit을 설치한다. Python 패키지와 CUDA Toolkit은 이미지에 들어간다.

## 1. 저장 위치와 마운트 구조

서버에 로그인한 사용자의 홈 디렉터리 아래에 다음과 같이 둔다.

```text
~/gr00t_ws/                         -> 컨테이너 /workspace/gr00t_ws/
├── Isaac-GR00T/                    # 이 저장소 전체, docker/ 파일 포함
├── models/                        # 원본 모델 / 별도로 가져온 체크포인트
├── datasets/                      # 학습 데이터
├── outputs/                       # 학습 체크포인트 / 평가 결과
├── logs/                          # 실행 로그 / W&B 실행 파일
├── cache/                         # Hugging Face, uv, Torch, Triton, CUDA 캐시
└── home/                          # 컨테이너 HOME, 셸 기록 및 사용자 설정
```

`~/gr00t_ws` 전체를 하나의 bind mount로 연결한다. 컨테이너에서 저장한 파일은
호스트에도 남고, 소스 변경은 즉시 반영된다. 컨테이너 기본 작업 디렉터리는
`/workspace/gr00t_ws/Isaac-GR00T`다. `~/gr00t_ws` 밖을 가리키는 심볼릭 링크는
컨테이너에서 접근되지 않으므로 실제 데이터도 이 디렉터리 아래에 둔다.

이미지의 Python 환경은 `/opt/gr00t-venv`에 둔다. 호스트의 `.venv`는 복사하거나
활성화하지 않는다. Docker 이미지/빌드 캐시 자체는 Docker가 관리하는 저장소에
남으며, `~/gr00t_ws` 마운트와는 별개다. 모델과 데이터 외에 이미지 빌드를 위한
디스크 여유도 확인한다.

## 2. 대상 서버: NVIDIA 드라이버 확인

아래부터 4절까지는 **대상 서버의 호스트 셸**에서 실행한다.

```bash
cat /etc/os-release
uname -m
nvidia-smi
df -h "$HOME" /var/lib
```

`nvidia-smi`가 GPU와 드라이버를 표시해야 한다. 실패하면 먼저 해당 GPU에서
CUDA 12.8을 지원하는 NVIDIA 드라이버를 설치/수정한다. 드라이버 설치는
[NVIDIA 드라이버 설치 안내](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)를 따른다.
드라이버가 정상인 서버에서는 호스트에 `uv`, PyTorch, CUDA Toolkit을 별도로 설치할 필요가 없다.

## 3. 대상 서버: Docker Engine + Compose 설치

Docker Engine과 `docker compose version`이 이미 정상 동작하면 이 설치는 건너뛴다.
아래는 새 Ubuntu 서버 기준이며, 배포판의 `docker.io` / `containerd` 등이 이미
설치되어 충돌하면 [Docker 공식 설치 안내](https://docs.docker.com/engine/install/ubuntu/)의
기존 패키지 정리 절차를 먼저 확인한다.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git git-lfs rsync
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

이후 **SSH 연결을 끊고 다시 로그인**해서 그룹 변경을 적용한다.
`docker` 그룹은 호스트 관리 권한을 가지므로 본인 계정에만 적용한다.
아래의 workspace 스크립트는 일반 사용자로 실행해야 해당 사용자의 UID/GID를 사용한다.

```bash
docker version
docker compose version
docker run --rm hello-world
```

Docker가 이미 설치된 서버에도 소스 전송용 도구는 필요하다.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git git-lfs rsync
```

## 4. 대상 서버: NVIDIA Container Toolkit 설치

[NVIDIA 공식 설치 안내](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)에
따라 Docker에 GPU 런타임을 연결한다. 마지막 Docker 재시작은 서버에서 실행 중인
다른 컨테이너에도 영향을 줄 수 있다. 이미 GPU 컨테이너가 동작하면 검증 명령만 실행한다.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all \
  nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

컨테이너에서도 GPU가 표시되어야 다음 단계로 진행할 수 있다.

## 5. 현재 머신 → 대상 서버: 현재 소스 복사

아직 커밋하지 않은 이번 Docker 파일까지 포함하려면 **현재 머신의 이 저장소에서**
`rsync`로 복사한다. `USER@SERVER_IP`만 대상 서버 계정/주소로 바꾼다.
현재 머신에도 `rsync`가 설치되어 있어야 한다.

```bash
cd /home/lim/rl_ws/Isaac-GR00T
GR00T_REMOTE='USER@SERVER_IP'

ssh "$GR00T_REMOTE" 'mkdir -p "$HOME/gr00t_ws/Isaac-GR00T"'

rsync -av --info=progress2 \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.cache/' \
  --exclude='/outputs/' \
  --exclude='/logs/' \
  --exclude='/wandb/' \
  --exclude='/models/' \
  --exclude='/checkpoints/' \
  ./ "$GR00T_REMOTE:gr00t_ws/Isaac-GR00T/"
```

이 명령은 `.git`, `uv.lock`, 서브모듈의 실제 소스, 저장소 내 작은 데모 데이터도
전송한다. 기존 대상 파일을 삭제하지는 않으며, 같은 경로의 파일은 현재 소스로 갱신한다.
별도 모델과 대용량 데이터는 아래처럼 옮긴다. 경로는 실제 위치로 바꾼다.

```bash
ssh "$GR00T_REMOTE" 'mkdir -p "$HOME/gr00t_ws/models" "$HOME/gr00t_ws/datasets/my_dataset"'
rsync -av --info=progress2 /path/to/models/ "$GR00T_REMOTE:gr00t_ws/models/"
rsync -av --info=progress2 /path/to/my_dataset/ "$GR00T_REMOTE:gr00t_ws/datasets/my_dataset/"
```

Git으로 설치하려면 **이번 파일들이 원격 브랜치에 커밋/푸시된 뒤** 대상 서버에서
아래 대체 명령을 사용한다. 현재 저장소의 `origin` 주소를 사용한 예시다.

```bash
mkdir -p "$HOME/gr00t_ws"
git clone --recurse-submodules https://github.com/achrxme-1838/Isaac-GR00T.git \
  "$HOME/gr00t_ws/Isaac-GR00T"
cd "$HOME/gr00t_ws/Isaac-GR00T"
# 기본 브랜치가 아니라면 여기서 Docker 파일이 포함된 브랜치/커밋으로 전환한다.
git submodule update --init --recursive
git lfs pull
```

## 6. 대상 서버: 초기화 → 빌드 → 실행 → 접속

이후 명령은 모두 **대상 서버의 호스트 셸**에서 실행한다.

```bash
cd "$HOME/gr00t_ws/Isaac-GR00T"

# 영구 저장 디렉터리를 현재 사용자 소유로 생성
bash docker/workspace.sh init

# 마운트 경로, UID/GID, GPU와 포트 설정 확인
bash docker/workspace.sh config

# 기존 docker/Dockerfile로 이미지 빌드 (최초 실행은 다운로드/빌드에 시간이 걸림)
bash docker/workspace.sh build

# 작업용 컨테이너를 백그라운드에서 계속 실행
bash docker/workspace.sh up -d
bash docker/workspace.sh ps

# 셸 접속
bash docker/workspace.sh exec gr00t bash
```

셸에서 `exit`해도 작업용 컨테이너는 유지된다. 빌드 시 인터넷 접근이 필요하며,
`~/gr00t_ws/Isaac-GR00T`와 다른 체크아웃에서 스크립트를 실행하면 경로 오류를 표시한다.

## 7. 대상 서버: GPU / Python / 마운트 검증

컨테이너 셸에서는 `exit`로 나온 후 호스트에서 실행한다.

```bash
cd "$HOME/gr00t_ws/Isaac-GR00T"
bash docker/workspace.sh exec gr00t nvidia-smi

bash docker/workspace.sh exec -T gr00t python - <<'PY'
import os
from pathlib import Path
import sys
import torch
import flash_attn
import torchcodec
import gr00t

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("UID/GID:", os.getuid(), os.getgid())
print("Source:", gr00t.__file__)
assert torch.cuda.is_available(), "컨테이너에서 CUDA를 사용할 수 없습니다."
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA tensor:", torch.ones(1, device="cuda").item())
Path("/workspace/gr00t_ws/outputs/mount-check.txt").write_text("mount OK\n")
PY

cat "$HOME/gr00t_ws/outputs/mount-check.txt"
ls -ln "$HOME/gr00t_ws/outputs/mount-check.txt"
id
```

파일 소유자의 숫자 UID/GID가 `id` 결과와 같아야 한다.
`python`은 이미지의 `/opt/gr00t-venv/bin/python`을 사용하고, `gr00t`는 마운트한
소스에서 import한다. Compose는 `UV_NO_SYNC=1`을 설정해서 `uv run`도 이미지의
환경을 그대로 사용한다([uv 환경 변수 안내](https://docs.astral.sh/uv/reference/environment/#uv_no_sync)).
기본 사용은 `python ...`으로 충분하다. 컨테이너에서 별도의 시뮬레이터 환경을
설치한다면 해당 예제 가이드를 따르고 `UV_NO_SYNC`, `UV_PROJECT_ENVIRONMENT`가
그 환경에 맞는지 확인한다.

## 8. 모델 다운로드와 추론 서버 실행

Hugging Face의 [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
접근 권한이 있는 계정으로 로그인한다. 저장소 README에 설명된 VLM 의존성이다.
로그인 정보/다운로드 캐시는 `~/gr00t_ws/cache/huggingface`에 저장된다.

```bash
cd "$HOME/gr00t_ws/Isaac-GR00T"
bash docker/workspace.sh exec gr00t hf auth login

bash docker/workspace.sh exec gr00t hf download nvidia/GR00T-N1.7-3B \
  --local-dir /workspace/gr00t_ws/models/GR00T-N1.7-3B

# 기본 모델에 포함된 DROID 태그로 실행하는 예시
bash docker/workspace.sh exec gr00t \
  python gr00t/eval/run_gr00t_server.py \
  --model-path /workspace/gr00t_ws/models/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --host 0.0.0.0 \
  --port 5555
```

마지막 명령은 foreground로 실행되며 `Ctrl+C`로 종료한다. 실제 로봇/데이터에
맞는 모델과 embodiment tag를 사용한다. `NEW_EMBODIMENT`는 해당 태그로 학습한
체크포인트에 사용한다. 이 포트는 HTTP 웹페이지가 아닌 GR00T의 ZeroMQ 정책 서버다.

기본 포트 매핑은 **서버의 `127.0.0.1:5555` → 컨테이너의 `5555`**다.
현재 머신에서 접근하려면 별도 터미널에 SSH 터널을 연다.

```bash
# 현재 머신에서 실행; 이 터미널을 열린 상태로 유지
ssh -N -L 5555:127.0.0.1:5555 USER@SERVER_IP
```

클라이언트의 서버 주소는 `127.0.0.1`, 포트는 `5555`로 설정한다.
로봇 PC가 서버 LAN IP로 직접 접속해야 하면 대상 서버에서 바인딩을 바꾸고
컨테이너를 재생성한 다음 추론 명령을 다시 실행한다.

```bash
# 대상 서버: LAN 접속이 필요한 경우에만 실행
export GR00T_BIND_IP=0.0.0.0
bash docker/workspace.sh up -d --force-recreate
```

이 설정은 모든 인터페이스로 포트를 열므로 접근 가능한 네트워크를 제한한다.
호스트 포트 충돌 시 `export GR00T_PORT=5556`으로 바꾸고 재생성한다.
컨테이너 안의 서버는 계속 `--port 5555`를 사용한다.

## 9. 학습 실행 예시

SO100 형식의 데이터가 `~/gr00t_ws/datasets/my_dataset`에 있다고 가정한 예시다.
실제 데이터에 맞춰 dataset 경로, modality config, embodiment tag를 함께 수정한다.
결과는 호스트의 `~/gr00t_ws/outputs/my_run`에 남는다.

```bash
cd "$HOME/gr00t_ws/Isaac-GR00T"
bash docker/workspace.sh exec \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NUM_GPUS=1 \
  -e USE_WANDB=0 \
  gr00t bash examples/finetune.sh \
  --base-model-path /workspace/gr00t_ws/models/GR00T-N1.7-3B \
  --dataset-path /workspace/gr00t_ws/datasets/my_dataset \
  --modality-config-path examples/SO100/so100_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /workspace/gr00t_ws/outputs/my_run
```

GPU 2개로 학습하려면 `CUDA_VISIBLE_DEVICES=0,1`, `NUM_GPUS=2`로 바꾼다.
배치 크기 등은 `GLOBAL_BATCH_SIZE`, `MAX_STEPS` 같은 환경 변수로 조정한다.
Compose는 모든 GPU를 컨테이너에 제공하며, 실행별 GPU 선택은
`CUDA_VISIBLE_DEVICES`로 한다. 컨테이너 자체에 제공할 GPU도 제한하려면
`compose.yaml`의 `count: all`을 `device_ids: ["0"]` 등으로 교체한다.
두 옵션은 함께 쓰지 않는다([Docker GPU 설정 안내](https://docs.docker.com/compose/how-tos/gpu-support/)).

장시간 학습은 대상 서버의 `tmux` 세션에서 위 명령을 실행하면 재접속하기 편하다.
컨테이너를 중지/재생성하면 그 안의 학습과 추론 프로세스도 종료된다.

## 10. 재접속, 업데이트, 종료

```bash
cd "$HOME/gr00t_ws/Isaac-GR00T"

# 실행 상태 및 셸 재접속
bash docker/workspace.sh ps
bash docker/workspace.sh exec gr00t bash

# 일시 중지 / 다시 시작 (호스트에서 실행)
bash docker/workspace.sh stop
bash docker/workspace.sh start

# 컨테이너와 Compose 네트워크 삭제; ~/gr00t_ws 파일은 유지
bash docker/workspace.sh down

# 다시 생성
bash docker/workspace.sh up -d
```

일반 Python 소스만 수정했다면 이미지를 다시 빌드할 필요가 없다. 실행 중인
Python 프로세스는 다시 시작한다. `pyproject.toml`, `uv.lock`, Dockerfile의
의존성을 바꿨다면 호스트에서 다음과 같이 빌드하고 컨테이너를 교체한다.

```bash
bash docker/workspace.sh build
bash docker/workspace.sh up -d --force-recreate
```

이미지의 가상환경은 root 소유이므로 일반 사용자 컨테이너 안에서 `uv sync`나
`pip install`로 직접 수정하지 않는다. 의존성 변경은 이미지에 반영한다.
`docker compose logs`에는 기본 대기 프로세스의 로그만 보이며,
`exec`로 실행한 학습/추론 로그는 해당 터미널에 출력된다.

설정을 바꾸려면 호스트 셸에서 `export`한 뒤 같은 셸에서 스크립트를 실행한다.
스크립트는 `.env`를 읽지 않는다. 기본 설정이면 별도 export가 필요 없다.

```bash
export GR00T_WS="$HOME/gr00t_ws"
export GR00T_IMAGE=gr00t:workspace
export GR00T_PROJECT_NAME=gr00t
export GR00T_BIND_IP=127.0.0.1
export GR00T_PORT=5555
```

`GR00T_WS`를 변경해도 저장소 위치는 반드시 `$GR00T_WS/Isaac-GR00T`여야 한다.
UID/GID는 매 실행 시 호스트 사용자에서 자동으로 가져온다.
