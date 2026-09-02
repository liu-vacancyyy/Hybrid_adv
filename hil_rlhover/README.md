# RLHover HIL

This folder runs the episode-720 hover policy as a host-side ONNX controller for PX4 Gazebo Classic HITL.

The policy trained in `gazebo_velocity_hover` outputs motor-level actions. PX4 Offboard does not accept this policy interface directly, so the runtime is a MAVLink proxy:

`Gazebo Classic HITL model <-> rlhover_proxy.py <-> PX4 FMU`

The proxy forwards all MAVLink traffic. When the configured RC switch is on, the FMU is armed, altitude is above the gate, and attitude/local-position messages are fresh, it replaces PX4 `HIL_ACTUATOR_CONTROLS` with ONNX motor commands. When the switch is off, it forwards PX4 actuator controls unchanged.

If the transmitter is unavailable, run the proxy with `--auto-rlhover`. In that mode PX4 keeps normal controller/PID actuator output during takeoff; after the proxy sees a stable hover for the configured hold time, it automatically starts the RL actuator override.

## 1. Install Python deps

```bash
cd /home/a/demo/Hybrid_adv
python -m pip install -r hil_rlhover/requirements.txt
```

`pymavlink` is also vendored under `/home/a/PX4-Autopilot/src/modules/mavlink/mavlink`; the proxy adds that path automatically, but serial links still need `pyserial`.

## 2. Export episode_720 to ONNX

Use the episode-720 checkpoint from the hover run you want to deploy:

```bash
cd /home/a/demo/Hybrid_adv
python hil_rlhover/export_velocity_hover_onnx.py \
  --ckpt /path/to/episode_720/actor_latest.ckpt \
  --onnx hil_rlhover/episode_720_velocity_hover.onnx
```

The expected interface is `obs=[1,27]`, `actions=[1,5]`, `rnn_states=[1,1,128]`.

## 3. Configure PX4

On the FMU shell or QGC MAVLink Console, use the HIL Standard VTOL QuadPlane airframe and paste these commands:

```text
param set SYS_HITL 1
param set COM_RC_IN_MODE 1
param set RC_MAP_AUX1 7
param set RC_MAP_RETURN_SW 0
param set RC_MAP_LOITER_SW 0
param set RC_MAP_OFFB_SW 0
param set CBRK_SUPPLY_CHK 894281
param save
```

The same commands are stored in `hil_rlhover/px4_rlhover_setup.px4sh` for reference, but QGC MAVLink Console cannot read a host filesystem path directly.

Important settings:

- `SYS_HITL=1`
- `RC_MAP_AUX1=7`, meaning RC channel 7 is the RLHover switch observed by the proxy
- `RC_MAP_RETURN_SW=0`, `RC_MAP_LOITER_SW=0`, and `RC_MAP_OFFB_SW=0`, avoiding a PX4 mode switch on the same channel while the proxy owns RLHover

If your transmitter currently uses CH7 for Return, either move RLHover to another free channel with `--rc-channel N`, or keep CH7 for RLHover and apply the setup script above so PX4 itself does not enter Return when the switch is high.

For this first integration, `RLHover` is implemented as "RC switch on + proxy override active". If you want `COM_FLTMODE` to show a separate `RLHover` enum, apply `hil_rlhover/px4_rlhover_flightmode_alias.patch` inside `/home/a/PX4-Autopilot`, rebuild/flash PX4, and then set `COM_FLTMODE6=17`. That alias maps to PX4 Offboard, so only use it after adding an Offboard keepalive/setpoint path or accepting that the proxy, not PX4 nav_state, owns the hover override.

## 4. Start Gazebo

Recommended one-command workflow for the real RC switch test:

```bash
cd /home/a/demo/Hybrid_adv
./hil_rlhover/start_rlhover_hitl.sh \
  --serial /dev/serial/by-id/usb-3D_Robotics_PX4_FMU_v5.x_0-if00 \
  --onnx hil_rlhover/episode_720_velocity_hover.onnx \
  --rc-channel 7 \
  --threshold 1600 \
  --min-altitude 1.0
```

This starts Gazebo with the HITL serial link disabled, then starts `rlhover_proxy.py` as the owner of the FMU serial port. Keep the RC switch low for normal PX4 PID takeoff. After the vehicle is in a stable multicopter hover above `--min-altitude`, switch RC channel 7 high to engage RLHover. The proxy will log `engaged` and replace PX4 `HIL_ACTUATOR_CONTROLS` with ONNX motor commands. Switch channel 7 low to return to PX4 PID actuator controls.

After this script is running, open QGC using UDP only. Do not enable the `Pixhawk-USB` QGC link because the proxy owns the serial device.

Manual two-terminal workflow, useful for debugging:

Terminal 1:

```bash
cd /home/a/demo/Hybrid_adv
bash hil_rlhover/start_gazebo_hitl_proxy.sh
```

This starts a temporary Standard VTOL HITL model with serial disabled. Do not use `standard_vtol_hitl` directly at the same time, because it would open `/dev/ttyACM0` itself.

## 5. Start the RLHover proxy

Terminal 2:

```bash
cd /home/a/demo/Hybrid_adv
python hil_rlhover/rlhover_proxy.py \
  --onnx hil_rlhover/episode_720_velocity_hover.onnx \
  --fcu serial:/dev/ttyACM0 \
  --baud 921600 \
  --gazebo-host 127.0.0.1 \
  --gazebo-port 14560 \
  --local-port 14557 \
  --rlhover-rc-channel 7 \
  --rlhover-threshold 1600 \
  --min-altitude 1.0
```

No-transmitter automatic switch:

```bash
cd /home/a/demo/Hybrid_adv
python hil_rlhover/rlhover_proxy.py \
  --onnx hil_rlhover/episode_720_velocity_hover.onnx \
  --fcu serial:/dev/ttyACM0 \
  --baud 921600 \
  --gazebo-host 127.0.0.1 \
  --gazebo-port 14560 \
  --local-port 14557 \
  --auto-takeoff \
  --takeoff-altitude 3.0 \
  --takeoff-delay 3.0 \
  --auto-rlhover \
  --auto-altitude 2.0 \
  --auto-vxy 0.6 \
  --auto-vz 0.35 \
  --auto-attitude-deg 12 \
  --auto-hold-seconds 1.5 \
  --min-altitude 1.0
```

Workflow:

1. Start Gazebo and the proxy.
2. With `--auto-takeoff`, the proxy sends PX4 arm/takeoff commands and PX4 does the initial takeoff.
3. In RC mode, switch RC channel 7 high after takeoff.
4. In `--auto-rlhover` mode, wait for the proxy to detect stable hover automatically.
5. The proxy logs `engaged` and begins sending ONNX motor controls.

For bench testing without the RC switch, add `--force-rlhover`; for motor-output bench tests only, add `--allow-disarmed`.
