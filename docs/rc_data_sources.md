# RC Command Data Sources

This project needs command traces in the form:

```text
t, vx_cmd, vz_cmd, heading_cmd
```

Optional but useful:

```text
yaw_rate_cmd, source_id
```

## Recommended Sources

### 1. PX4 Flight Review Public Logs

- URL: https://logs.px4.io/browse
- Format: PX4 `.ulg`
- Best fields:
  - `manual_control_setpoint.roll/pitch/yaw/throttle`
  - `input_rc.values[*]`
  - `trajectory_setpoint.velocity[0/1/2]`
  - `trajectory_setpoint.yaw`
  - `vehicle_local_position_setpoint.vx/vy/vz/yaw`
  - `vehicle_local_position.vx/vy/vz/heading`
- Use for:
  - real manual flight logs if `manual_control_setpoint` or `input_rc` has enough samples;
  - automatic/offboard setpoint traces if manual RC is absent.

This is the highest-priority source because the logs use the same PX4 ULog format
as the local SITL log.

### 2. PX4 pyulog

- URL: https://github.com/PX4/pyulog
- Purpose: parse `.ulg` and export topics to CSV.
- Local status:
  - installed in `/home/a/anaconda3/envs/Neuralplane`
  - used by `scripts/eval/extract_px4_rc_data.py`

### 3. PX4 Flight Review Tooling

- URL: https://github.com/PX4/flight_review
- Purpose: inspect public logs and identify logs that contain manual control topics.
- Use for:
  - quickly checking whether a log has `manual_control_setpoint` / `input_rc`
  - avoiding automatic-only logs.

### 4. ETHZ-ASL Data-Driven Dynamics

- URL: https://github.com/ethz-asl/data-driven-dynamics
- Purpose: data-driven UAV dynamics from PX4-style logs.
- Use for:
  - workflow reference;
  - dynamics/state datasets.
- Limitation:
  - not primarily an RC stick dataset.

### 5. Local PX4 SITL Logs

Current local log:

```text
/home/a/PX4-Autopilot/build/px4_sitl_default/rootfs/log/2026-03-31/07_46_00.ulg
```

Extracted fields:

```text
vehicle_local_position: 4445 samples
trajectory_setpoint: 177 samples
vehicle_local_position_setpoint: 351 samples
manual_control_setpoint: ignored, only 1 sample
input_rc: 0 samples
```

Conclusion:

```text
This log is useful as an automatic setpoint/reference trace, not as real manual RC data.
```

## Generated Local Datasets

Created by:

```bash
/home/a/anaconda3/envs/Neuralplane/bin/python scripts/eval/extract_px4_rc_data.py
/home/a/anaconda3/envs/Neuralplane/bin/python scripts/eval/build_rc_command_dataset.py
```

Outputs:

```text
renders/result/px4_rc_extract/07_46_00_rc_commands.csv
renders/result/px4_rc_extract/07_46_00_rc_commands.npz

data/rc_commands/rc_px4_reference.csv
data/rc_commands/rc_px4_reference.npz
data/rc_commands/rc_humanlike_synthetic.csv
data/rc_commands/rc_humanlike_synthetic.npz
data/rc_commands/rc_mixed_commands.csv
data/rc_commands/rc_mixed_commands.npz
```

Dataset meanings:

```text
rc_px4_reference:
  cleaned PX4 setpoint trace from the local SITL log.

rc_humanlike_synthetic:
  generated pilot-like command traces using piecewise stick targets,
  low-pass smoothing, and bounded yaw-rate integration.

rc_mixed_commands:
  concatenation of PX4 reference + synthetic commands.
```

Recommended training starting point:

```text
data/rc_commands/rc_mixed_commands.npz
```

Recommended evaluation split:

```text
use source_id == 0 for PX4-reference behaviour checks;
use held-out synthetic episodes for broad RC command tracking.
```
