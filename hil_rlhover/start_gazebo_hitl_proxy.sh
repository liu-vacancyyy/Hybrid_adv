#!/usr/bin/env bash
# Launch Gazebo Classic with a Standard VTOL HITL model wired for the RLHover proxy.
#
# This intentionally does not open the FCU serial device. The proxy owns the
# serial link and forwards MAVLink between Gazebo and PX4.

set -euo pipefail

PX4_DIR="${PX4_DIR:-/home/a/PX4-Autopilot}"
GAZEBO_ROOT="${PX4_DIR}/Tools/simulation/gazebo-classic/sitl_gazebo-classic"
PX4_BUILD="${PX4_BUILD:-${PX4_DIR}/build/px4_sitl_default}"
GAZEBO_BUILD="${GAZEBO_BUILD:-${PX4_BUILD}/build_gazebo-classic}"

WORK_DIR="${WORK_DIR:-/tmp/rlhover_hitl_gazebo}"
MODEL_NAME="standard_vtol_hitl_proxy"
GAZEBO_MAVLINK_UDP_PORT="${GAZEBO_MAVLINK_UDP_PORT:-14560}"
GAZEBO_MAVLINK_TCP_PORT="${GAZEBO_MAVLINK_TCP_PORT:-4560}"

if [[ ! -d "${GAZEBO_ROOT}" ]]; then
    echo "Gazebo Classic tree not found: ${GAZEBO_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${GAZEBO_BUILD}" ]]; then
    echo "Gazebo Classic build not found: ${GAZEBO_BUILD}" >&2
    echo "Build PX4 SITL/Gazebo Classic first, then retry." >&2
    exit 1
fi

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/models/${MODEL_NAME}"
cp -a "${GAZEBO_ROOT}/models/standard_vtol_hitl/." "${WORK_DIR}/models/${MODEL_NAME}/"

python3 - "$WORK_DIR/models/$MODEL_NAME/standard_vtol_hitl.sdf" "$MODEL_NAME" "$GAZEBO_MAVLINK_UDP_PORT" "$GAZEBO_MAVLINK_TCP_PORT" <<'PY'
from pathlib import Path
import sys

sdf = Path(sys.argv[1])
model_name = sys.argv[2]
udp_port = sys.argv[3]
tcp_port = sys.argv[4]
text = sdf.read_text()
text = text.replace("<model name='standard_vtol_hitl'>", f"<model name='{model_name}'>")
text = text.replace("<serialEnabled>1</serialEnabled>", "<serialEnabled>0</serialEnabled>")
text = text.replace("<use_tcp>1</use_tcp>", "<use_tcp>0</use_tcp>")
text = text.replace("<mavlink_udp_port>14560</mavlink_udp_port>", f"<mavlink_udp_port>{udp_port}</mavlink_udp_port>")
text = text.replace("<mavlink_tcp_port>4560</mavlink_tcp_port>", f"<mavlink_tcp_port>{tcp_port}</mavlink_tcp_port>")
text = text.replace("standard_vtol_hitl/", f"{model_name}/")
sdf.write_text(text)
PY

python3 - "$WORK_DIR/models/$MODEL_NAME/model.config" "$MODEL_NAME" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
model_name = sys.argv[2]
text = path.read_text()
text = text.replace("<name>Standard VTOL HITL</name>", "<name>Standard VTOL HITL Proxy</name>")
text = text.replace("<sdf version='1.5'>standard_vtol_hitl.sdf</sdf>", "<sdf version='1.5'>standard_vtol_hitl.sdf</sdf>")
path.write_text(text)
PY

cat > "${WORK_DIR}/rlhover_hitl.world" <<EOF_WORLD
<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="rlhover_hitl_world">
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
        <solver><type>quick</type><iters>10</iters><sor>1.3</sor><use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling></solver>
        <constraints><cfm>0</cfm><erp>0.2</erp><contact_max_correcting_vel>100</contact_max_correcting_vel><contact_surface_layer>0.001</contact_surface_layer></constraints>
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

echo "Gazebo model: ${WORK_DIR}/models/${MODEL_NAME}"
echo "Gazebo world: ${WORK_DIR}/rlhover_hitl.world"
echo "MAVLink HITL UDP port: ${GAZEBO_MAVLINK_UDP_PORT}"

exec gazebo --verbose "${WORK_DIR}/rlhover_hitl.world"
