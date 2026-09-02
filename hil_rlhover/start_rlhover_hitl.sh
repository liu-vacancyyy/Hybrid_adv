#!/usr/bin/env bash
# Start PX4 Standard VTOL HITL through the RLHover MAVLink proxy.
#
# Control path:
#   RC switch low:  PX4 FMU PID -> HIL_ACTUATOR_CONTROLS -> proxy -> Gazebo
#   RC switch high: ONNX hover policy -> HIL_ACTUATOR_CONTROLS -> Gazebo
#
# Gazebo never opens the FMU serial device in this mode. The proxy owns the
# serial link, forwards all MAVLink traffic, and only replaces actuator controls
# when the RLHover RC switch and safety gates are active.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)

PX4_DIR="${PX4_DIR:-/home/a/PX4-Autopilot}"
GAZEBO_ROOT="${PX4_DIR}/Tools/simulation/gazebo-classic/sitl_gazebo-classic"
PX4_BUILD="${PX4_BUILD:-${PX4_DIR}/build/px4_sitl_default}"
GAZEBO_BUILD="${GAZEBO_BUILD:-${PX4_BUILD}/build_gazebo-classic}"

ONNX_PATH="${ONNX_PATH:-${script_dir}/episode_720_velocity_hover.onnx}"
SERIAL_DEVICE="${SERIAL_DEVICE:-${PX4_HITL_SERIAL_DEVICE:-}}"
BAUD_RATE="${BAUD_RATE:-${PX4_HITL_BAUD_RATE:-921600}}"
PYTHON_BIN="${PYTHON_BIN:-}"

GAZEBO_HOST="${GAZEBO_HOST:-127.0.0.1}"
GAZEBO_MAVLINK_UDP_PORT="${GAZEBO_MAVLINK_UDP_PORT:-14560}"
PROXY_LOCAL_PORT="${PROXY_LOCAL_PORT:-14557}"
GAZEBO_MAVLINK_TCP_PORT="${GAZEBO_MAVLINK_TCP_PORT:-4560}"

RLHOVER_RC_CHANNEL="${RLHOVER_RC_CHANNEL:-7}"
RLHOVER_THRESHOLD="${RLHOVER_THRESHOLD:-1600}"
MIN_ALTITUDE="${MIN_ALTITUDE:-1.0}"
RATE_HZ="${RATE_HZ:-50}"
PRINT_RATE="${PRINT_RATE:-1.0}"
WORK_DIR="${WORK_DIR:-}"
HEADLESS="${HEADLESS:-0}"

AUTO_TAKEOFF="${AUTO_TAKEOFF:-0}"
TAKEOFF_ALTITUDE="${TAKEOFF_ALTITUDE:-3.0}"
TAKEOFF_DELAY="${TAKEOFF_DELAY:-3.0}"
AUTO_RLHOVER="${AUTO_RLHOVER:-0}"
AUTO_ALTITUDE="${AUTO_ALTITUDE:-2.0}"
AUTO_VXY="${AUTO_VXY:-0.6}"
AUTO_VZ="${AUTO_VZ:-0.35}"
AUTO_ATTITUDE_DEG="${AUTO_ATTITUDE_DEG:-12}"
AUTO_HOLD_SECONDS="${AUTO_HOLD_SECONDS:-1.5}"

usage() {
	cat <<'EOF'
Usage: hil_rlhover/start_rlhover_hitl.sh [options]

Starts Gazebo Classic Standard VTOL HITL plus the RLHover ONNX proxy.

Options:
  --serial DEVICE       Flight-controller MAVLink serial device. Auto-detected by default.
  --baud RATE           Serial baud rate (default: 921600).
  --onnx PATH           ONNX hover policy (default: hil_rlhover/episode_720_velocity_hover.onnx).
  --rc-channel N        1-based RC channel that enables RLHover (default: 7).
  --threshold PWM       PWM threshold for RLHover switch high (default: 1600).
  --min-altitude M      Minimum altitude before RL takeover (default: 1.0).
  --headless            Run gzserver only.
  --auto-takeoff        Ask PX4 to arm/take off before optional auto RLHover.
  --auto-rlhover        Auto-engage RLHover after stable hover gates pass.
  --help                Show this help.

Environment variables mirror the options, for example:
  SERIAL_DEVICE=/dev/serial/by-id/... RLHOVER_RC_CHANNEL=7 ONNX_PATH=...

QGC must not open the USB serial port. Start QGC after this script and use UDP.
EOF
}

need_value() {
	if [[ $# -lt 2 || -z "${2:-}" ]]; then
		echo "ERROR: $1 requires a value" >&2
		exit 2
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--serial)
		need_value "$@"
		SERIAL_DEVICE="$2"
		shift 2
		;;
	--baud)
		need_value "$@"
		BAUD_RATE="$2"
		shift 2
		;;
	--onnx)
		need_value "$@"
		ONNX_PATH="$2"
		shift 2
		;;
	--rc-channel)
		need_value "$@"
		RLHOVER_RC_CHANNEL="$2"
		shift 2
		;;
	--threshold)
		need_value "$@"
		RLHOVER_THRESHOLD="$2"
		shift 2
		;;
	--min-altitude)
		need_value "$@"
		MIN_ALTITUDE="$2"
		shift 2
		;;
	--headless)
		HEADLESS=1
		shift
		;;
	--auto-takeoff)
		AUTO_TAKEOFF=1
		shift
		;;
	--auto-rlhover)
		AUTO_RLHOVER=1
		shift
		;;
	-h|--help)
		usage
		exit 0
		;;
	*)
		echo "ERROR: unknown option: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

if [[ ! "$BAUD_RATE" =~ ^[0-9]+$ ]] || (( BAUD_RATE < 9600 || BAUD_RATE > 3000000 )); then
	echo "ERROR: invalid baud rate: ${BAUD_RATE}" >&2
	exit 2
fi

if [[ ! "$RLHOVER_RC_CHANNEL" =~ ^[0-9]+$ ]] || (( RLHOVER_RC_CHANNEL < 1 || RLHOVER_RC_CHANNEL > 18 )); then
	echo "ERROR: --rc-channel must be 1..18" >&2
	exit 2
fi

if [[ ! "$RLHOVER_THRESHOLD" =~ ^[0-9]+$ ]] || (( RLHOVER_THRESHOLD < 900 || RLHOVER_THRESHOLD > 2200 )); then
	echo "ERROR: --threshold must be a PWM value, usually 1000..2000" >&2
	exit 2
fi

if [[ ! -f "$ONNX_PATH" ]]; then
	echo "ERROR: ONNX policy not found: ${ONNX_PATH}" >&2
	echo "Export it first, for example:" >&2
	echo "  python hil_rlhover/export_velocity_hover_onnx.py --ckpt ckpt/hover_720/actor_latest.ckpt --onnx hil_rlhover/episode_720_velocity_hover.onnx" >&2
	exit 1
fi

detect_serial_device() {
	local candidate
	local -a candidates=()

	if [[ -d /dev/serial/by-id ]]; then
		while IFS= read -r candidate; do
			candidates+=("$candidate")
		done < <(find /dev/serial/by-id -maxdepth 1 -type l -print | sort)
	fi

	if (( ${#candidates[@]} == 0 )); then
		while IFS= read -r candidate; do
			candidates+=("$candidate")
		done < <(compgen -G '/dev/ttyACM*' | sort || true)
	fi

	if (( ${#candidates[@]} == 0 )); then
		while IFS= read -r candidate; do
			candidates+=("$candidate")
		done < <(compgen -G '/dev/ttyUSB*' | sort || true)
	fi

	if (( ${#candidates[@]} > 0 )); then
		if (( ${#candidates[@]} > 1 )); then
			echo "ERROR: multiple serial devices found; select the FMU with --serial:" >&2
			printf '  %s\n' "${candidates[@]}" >&2
			return 1
		fi

		printf '%s\n' "${candidates[0]}"
	fi
}

if [[ -z "$SERIAL_DEVICE" ]]; then
	SERIAL_DEVICE=$(detect_serial_device)
fi

if [[ -z "$SERIAL_DEVICE" ]]; then
	echo "ERROR: no flight-controller serial device found; pass --serial /dev/..." >&2
	exit 1
fi

if [[ ! "$SERIAL_DEVICE" =~ ^/dev/[A-Za-z0-9._/+:-]+$ ]]; then
	echo "ERROR: serial device must be an absolute path below /dev: ${SERIAL_DEVICE}" >&2
	exit 2
fi

if [[ ! -e "$SERIAL_DEVICE" ]]; then
	echo "ERROR: serial device does not exist: ${SERIAL_DEVICE}" >&2
	exit 1
fi

if [[ ! -r "$SERIAL_DEVICE" || ! -w "$SERIAL_DEVICE" ]]; then
	echo "ERROR: no read/write permission for ${SERIAL_DEVICE}; add the user to dialout and log in again" >&2
	exit 1
fi

if command -v fuser >/dev/null 2>&1 && fuser "$SERIAL_DEVICE" >/dev/null 2>&1; then
	echo "ERROR: ${SERIAL_DEVICE} is already in use. Close QGroundControl and old Gazebo/proxy processes first." >&2
	exit 1
fi

if pgrep -f 'QGroundControl(.AppImage)?' >/dev/null 2>&1; then
	echo "ERROR: QGroundControl is running. Close it, start RLHover HITL, then reopen QGC using UDP only." >&2
	exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
	if command -v python >/dev/null 2>&1; then
		PYTHON_BIN=python
	else
		PYTHON_BIN=python3
	fi
fi

"$PYTHON_BIN" - <<'PY'
import sys
missing = []
for module in ("numpy", "onnxruntime", "serial"):
    try:
        __import__(module)
    except Exception:
        missing.append(module)
if missing:
    print("ERROR: missing Python modules: " + ", ".join(missing), file=sys.stderr)
    print("Install inside the active environment:", file=sys.stderr)
    print("  python -m pip install -r hil_rlhover/requirements.txt", file=sys.stderr)
    sys.exit(1)
PY

for required_path in \
	"${GAZEBO_ROOT}/models/standard_vtol_hitl/model.config" \
	"${GAZEBO_ROOT}/models/standard_vtol_hitl/standard_vtol_hitl.sdf" \
	"${GAZEBO_ROOT}/worlds/hitl_standard_vtol.world"; do
	if [[ ! -f "$required_path" ]]; then
		echo "ERROR: required HITL asset is missing: ${required_path}" >&2
		exit 1
	fi
done

if [[ ! -f "${GAZEBO_BUILD}/libgazebo_mavlink_interface.so" ]]; then
	echo "ERROR: Gazebo Classic MAVLink plugin is missing: ${GAZEBO_BUILD}/libgazebo_mavlink_interface.so" >&2
	echo "Run: DONT_RUN=1 make -C ${PX4_DIR} px4_sitl_default gazebo-classic" >&2
	exit 1
fi

runtime_dir="${WORK_DIR}"
if [[ -z "$runtime_dir" ]]; then
	runtime_dir=$(mktemp -d /tmp/px4-rlhover-hitl.XXXXXX)
else
	rm -rf -- "$runtime_dir"
	mkdir -p "$runtime_dir"
fi

gazebo_pid=""
cleanup() {
	if [[ -n "$gazebo_pid" ]] && kill -0 "$gazebo_pid" >/dev/null 2>&1; then
		kill "$gazebo_pid" >/dev/null 2>&1 || true
		wait "$gazebo_pid" >/dev/null 2>&1 || true
	fi

	if [[ -z "${WORK_DIR:-}" && -n "$runtime_dir" && -d "$runtime_dir" && "$runtime_dir" == /tmp/px4-rlhover-hitl.* ]]; then
		rm -rf -- "$runtime_dir"
	fi
}
trap cleanup EXIT INT TERM

runtime_model="${runtime_dir}/models/standard_vtol_hitl"
mkdir -p "$runtime_model"
cp "${GAZEBO_ROOT}/models/standard_vtol_hitl/model.config" "$runtime_model/model.config"

# Generate a disposable model that talks MAVLink UDP to the proxy instead of
# opening the flight controller serial device directly.
sed \
	-e 's#<serialEnabled>1</serialEnabled>#<serialEnabled>0</serialEnabled>#' \
	-e 's#<use_tcp>1</use_tcp>#<use_tcp>0</use_tcp>#' \
	-e "s#<mavlink_udp_port>14560</mavlink_udp_port>#<mavlink_udp_port>${GAZEBO_MAVLINK_UDP_PORT}</mavlink_udp_port>#" \
	-e "s#<mavlink_tcp_port>4560</mavlink_tcp_port>#<mavlink_tcp_port>${GAZEBO_MAVLINK_TCP_PORT}</mavlink_tcp_port>#" \
	"${GAZEBO_ROOT}/models/standard_vtol_hitl/standard_vtol_hitl.sdf" > "$runtime_model/standard_vtol_hitl.sdf"

if ! grep -Fq '<serialEnabled>0</serialEnabled>' "$runtime_model/standard_vtol_hitl.sdf"; then
	echo "ERROR: failed to disable Gazebo serial mode for RLHover proxy" >&2
	exit 1
fi

if ! grep -Fq '<hil_mode>1</hil_mode>' "$runtime_model/standard_vtol_hitl.sdf"; then
	echo "ERROR: runtime model is not configured for HITL" >&2
	exit 1
fi

runtime_world="${runtime_dir}/hitl_standard_vtol_rlhover.world"
sed \
	-e "s#<uri>model://standard_vtol_hitl</uri>#<uri>model://standard_vtol_hitl</uri>#" \
	"${GAZEBO_ROOT}/worlds/hitl_standard_vtol.world" > "$runtime_world"

export GAZEBO_MODEL_PATH="${runtime_dir}/models${GAZEBO_MODEL_PATH:+:${GAZEBO_MODEL_PATH}}"
export GAZEBO_PLUGIN_PATH="${GAZEBO_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# shellcheck source=/dev/null
source "${PX4_DIR}/Tools/simulation/gazebo-classic/setup_gazebo.bash" "$PX4_DIR" "$PX4_BUILD"

echo "PX4 VTOL RLHover HITL"
echo "  serial:     ${SERIAL_DEVICE} @ ${BAUD_RATE}"
echo "  onnx:       ${ONNX_PATH}"
echo "  RC switch:  channel ${RLHOVER_RC_CHANNEL} >= ${RLHOVER_THRESHOLD}"
echo "  safety:     armed, altitude >= ${MIN_ALTITUDE}m, fresh attitude/local position"
echo "  Gazebo UDP: ${GAZEBO_HOST}:${GAZEBO_MAVLINK_UDP_PORT} <- proxy local ${PROXY_LOCAL_PORT}"
echo "  QGC:        start it after this script; UDP only, no Pixhawk-USB link"

if [[ "$HEADLESS" == "1" ]]; then
	gzserver --verbose "$runtime_world" &
else
	gazebo --verbose "$runtime_world" &
fi
gazebo_pid=$!

sleep 2
if ! kill -0 "$gazebo_pid" >/dev/null 2>&1; then
	echo "ERROR: Gazebo exited before the proxy could start" >&2
	exit 1
fi

proxy_args=(
	"${script_dir}/rlhover_proxy.py"
	--onnx "$ONNX_PATH"
	--fcu "serial:${SERIAL_DEVICE}"
	--baud "$BAUD_RATE"
	--gazebo-host "$GAZEBO_HOST"
	--gazebo-port "$GAZEBO_MAVLINK_UDP_PORT"
	--local-port "$PROXY_LOCAL_PORT"
	--rate-hz "$RATE_HZ"
	--print-rate "$PRINT_RATE"
	--rlhover-rc-channel "$RLHOVER_RC_CHANNEL"
	--rlhover-threshold "$RLHOVER_THRESHOLD"
	--min-altitude "$MIN_ALTITUDE"
)

if [[ "$AUTO_TAKEOFF" == "1" ]]; then
	proxy_args+=(--auto-takeoff --takeoff-altitude "$TAKEOFF_ALTITUDE" --takeoff-delay "$TAKEOFF_DELAY")
fi

if [[ "$AUTO_RLHOVER" == "1" ]]; then
	proxy_args+=(
		--auto-rlhover
		--auto-altitude "$AUTO_ALTITUDE"
		--auto-vxy "$AUTO_VXY"
		--auto-vz "$AUTO_VZ"
		--auto-attitude-deg "$AUTO_ATTITUDE_DEG"
		--auto-hold-seconds "$AUTO_HOLD_SECONDS"
	)
fi

"$PYTHON_BIN" "${proxy_args[@]}"
