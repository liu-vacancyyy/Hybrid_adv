#!/usr/bin/env bash
# Launch PX4 Gazebo Classic HITL with the stock PX4 PID controller path.
#
# Topology:
#   Gazebo Classic standard_vtol_hitl <serial> PX4 FMU
#
# No ONNX runtime, no rlhover_proxy.py, and no actuator override are used here.
# Gazebo owns the serial port and PX4 produces the actuator controls.

set -euo pipefail

PX4_DIR="${PX4_DIR:-/home/a/PX4-Autopilot}"
GAZEBO_ROOT="${PX4_DIR}/Tools/simulation/gazebo-classic/sitl_gazebo-classic"
PX4_BUILD="${PX4_BUILD:-${PX4_DIR}/build/px4_sitl_default}"
GAZEBO_BUILD="${GAZEBO_BUILD:-${PX4_BUILD}/build_gazebo-classic}"

SERIAL_DEVICE="${SERIAL_DEVICE:-auto}"
BAUD_RATE="${BAUD_RATE:-921600}"
WORK_DIR="${WORK_DIR:-/tmp/px4_pid_hitl_gazebo}"
MODEL_NAME="${MODEL_NAME:-standard_vtol_hitl_pid}"
WORLD_NAME="${WORLD_NAME:-px4_pid_hitl.world}"

if [[ ! -d "${GAZEBO_ROOT}" ]]; then
    echo "Gazebo Classic tree not found: ${GAZEBO_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${GAZEBO_BUILD}" ]]; then
    echo "Gazebo Classic build not found: ${GAZEBO_BUILD}" >&2
    echo "Build PX4 SITL/Gazebo Classic first, then retry." >&2
    exit 1
fi

if [[ "${SERIAL_DEVICE}" = "auto" ]]; then
    SERIAL_DEVICE=""

    for dev in /dev/serial/by-id/*PX4* /dev/serial/by-id/*FMU* /dev/serial/by-id/*3D_Robotics*; do
        if [[ -e "${dev}" ]]; then
            SERIAL_DEVICE="$(readlink -f "${dev}")"
            break
        fi
    done

    if [[ -z "${SERIAL_DEVICE}" ]]; then
        for dev in /dev/ttyACM* /dev/ttyUSB*; do
            if [[ -e "${dev}" ]]; then
                SERIAL_DEVICE="${dev}"
                break
            fi
        done
    fi

    if [[ -z "${SERIAL_DEVICE}" ]]; then
        SERIAL_DEVICE="/dev/ttyACM0"
    fi
fi

if [[ "${SKIP_SERIAL_CHECK:-0}" != "1" ]]; then
    if [[ ! -e "${SERIAL_DEVICE}" ]]; then
        echo "Serial device not found: ${SERIAL_DEVICE}" >&2
        echo "Available serial devices:" >&2
        ls -l /dev/serial/by-id/* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null >&2 || true
        echo "Set SERIAL_DEVICE=/dev/ttyACM1 if your FMU enumerated differently." >&2
        exit 1
    fi

    if [[ ! -r "${SERIAL_DEVICE}" || ! -w "${SERIAL_DEVICE}" ]]; then
        echo "Serial device is not readable/writable: ${SERIAL_DEVICE}" >&2
        echo "Check dialout permission or run from a user with access to the FMU serial port." >&2
        exit 1
    fi
fi

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/models/${MODEL_NAME}"
cp -a "${GAZEBO_ROOT}/models/standard_vtol_hitl/." "${WORK_DIR}/models/${MODEL_NAME}/"

python3 - "${WORK_DIR}/models/${MODEL_NAME}/standard_vtol_hitl.sdf" "${MODEL_NAME}" "${SERIAL_DEVICE}" "${BAUD_RATE}" <<'PY'
from pathlib import Path
import sys

sdf = Path(sys.argv[1])
model_name = sys.argv[2]
serial_device = sys.argv[3]
baud_rate = sys.argv[4]

text = sdf.read_text()
text = text.replace("<model name='standard_vtol_hitl'>", f"<model name='{model_name}'>")
text = text.replace("<serialEnabled>0</serialEnabled>", "<serialEnabled>1</serialEnabled>")
text = text.replace("<serialEnabled>false</serialEnabled>", "<serialEnabled>1</serialEnabled>")
text = text.replace("<serialDevice>/dev/ttyACM0</serialDevice>", f"<serialDevice>{serial_device}</serialDevice>")
text = text.replace("<baudRate>921600</baudRate>", f"<baudRate>{baud_rate}</baudRate>")
text = text.replace("standard_vtol_hitl/", f"{model_name}/")
sdf.write_text(text)
PY

python3 - "${WORK_DIR}/models/${MODEL_NAME}/model.config" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("<name>Standard VTOL HITL</name>", "<name>Standard VTOL HITL PID</name>")
path.write_text(text)
PY

cat > "${WORK_DIR}/${WORLD_NAME}" <<EOF_WORLD
<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="px4_pid_hitl_world">
    <scene>
      <ambient>0.7 0.7 0.7 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>false</shadows>
    </scene>

    <gui fullscreen='0'>
      <camera name='user_camera'>
        <pose>-12.0 -3.0 10.0 0 0.5 0.2</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>

    <include><uri>model://sun</uri></include>

    <physics name='default_physics' default='0' type='ode'>
      <gravity>0 0 -9.8066</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
          <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
      <magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>
    </physics>

    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://asphalt_plane</uri></include>
    <include>
      <uri>model://${MODEL_NAME}</uri>
      <pose>1.01 0.98 0.83 0 0 1.14</pose>
    </include>
  </world>
</sdf>
EOF_WORLD

export GAZEBO_MODEL_PATH="${WORK_DIR}/models:${GAZEBO_ROOT}/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_PLUGIN_PATH="${GAZEBO_BUILD}:${GAZEBO_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="${GAZEBO_BUILD}:${LD_LIBRARY_PATH:-}"

echo "PX4 PID HITL Gazebo model: ${WORK_DIR}/models/${MODEL_NAME}"
echo "PX4 PID HITL Gazebo world: ${WORK_DIR}/${WORLD_NAME}"
echo "FCU serial: ${SERIAL_DEVICE} @ ${BAUD_RATE}"
echo "Controller path: PX4 FMU PID -> HIL_ACTUATOR_CONTROLS -> Gazebo"
echo "QGC/MAVSDK UDP stays available through the Gazebo MAVLink plugin on 14550/14540."

exec gazebo --verbose "${WORK_DIR}/${WORLD_NAME}"
