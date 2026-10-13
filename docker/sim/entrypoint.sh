#!/bin/bash
set -e

ISAAC_SIM=/workspace/isaaclab/_isaac_sim

export CARB_APP_PATH=$ISAAC_SIM/kit
export ISAAC_PATH=$ISAAC_SIM
export EXP_PATH=$ISAAC_SIM/apps
source ${ISAAC_SIM}/setup_python_env.sh
source /root/env 2>/dev/null

cat > /usr/local/bin/python << 'WRAPPER'
#!/bin/bash
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
WRAPPER
chmod +x /usr/local/bin/python

python -m pip install -e /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/

# OMX-L(Dynamixel XL330) 리더용 SDK. 컨테이너를 새로 만들 때마다 사라지므로 여기서 자동 설치한다(이미 있으면 건너뜀).
# --no-deps: Isaac Lab 이 심볼릭 링크로 쓰는 패키지(pyserial 등)를 건드리지 않기 위함.
if ! python -m pip show dynamixel-sdk >/dev/null 2>&1; then
  python -m pip install --no-deps "dynamixel-sdk>=3.7.31,<3.9.0" \
    || echo "[entrypoint][WARN] dynamixel-sdk 설치 실패(네트워크 확인). OMX 리더를 쓰려면 수동 설치 필요."
fi

exec "$@"
