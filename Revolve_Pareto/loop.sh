#!/usr/bin/env bash
set -euo pipefail

# ===== CONFIG =====
CARLA_DIR="$HOME/Downloads/carla"
MAP="/Game/Carla/Maps/Town01"
RESX=1280
RESY=720
QUALITY="High"
WAIT_BOOT=18
EXTRA_WAIT=18
COOL_DOWN=10

EPISODES=10
TOTAL=16           # number of individuals per generation
PYTHON="python3"
MAIN_PY="main.py"
export ROOT_PATH="..."
export GEN_ID=0

# ===== FUNCTIONS =====
wait_for_rpc() {  
  local t=0
  while ! (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null; do
    sleep 1; t=$((t+1))
    (( t >= WAIT_BOOT )) && break
  done
  exec 3>&- || true
}

start_carla() {
  pushd "$CARLA_DIR" >/dev/null
  echo "[CARLA] Launching simulator..."
  ./CarlaUE4.sh -ResX="$RESX" -ResY="$RESY" -quality-level="$QUALITY" "$MAP" >/dev/null 2>&1 &
  CARLA_PID=$!
  popd >/dev/null
  wait_for_rpc
  echo "[CARLA] RPC ready. Waiting ${EXTRA_WAIT}s for navmesh/assets..."
  sleep "$EXTRA_WAIT"
}

stop_carla() {
  local pid="$1"
  echo "[CARLA] Stopping simulator PID=$pid..."
  kill "$pid" >/dev/null 2>&1 || true
  sleep 2
  kill -9 "$pid" >/dev/null 2>&1 || true
  echo "[CARLA] Stopped."
}

trap '[[ ${CARLA_PID:-} ]] && stop_carla "$CARLA_PID"' EXIT INT TERM

# ===== MAIN LOOP =====
#
for ((COUNTER=0; COUNTER<$TOTAL; COUNTER++)); do
#for COUNTER in 5 10 13   ; do

  echo "===== Starting Individual $COUNTER ====="

  export COUNTER=$COUNTER
  export EPISODES=$EPISODES

  start_carla
  set +e
  $PYTHON "$MAIN_PY"
  rc=$?
  set -e
  echo "[main.py exit code] $rc"
  stop_carla "$CARLA_PID"; unset CARLA_PID
  echo "[WAIT] Cooling down ${COOL_DOWN}s before next run..."
  sleep "$COOL_DOWN"
done

echo "=== All 16 individuals complete ==="
