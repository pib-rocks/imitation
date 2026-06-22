# Hand tracking for pib imitation

Runs Mediapipe hand models on a Luxonis OAK camera (Edge mode) and maps detected hand poses to robot joint commands via ROS2.

## Prerequisites

- OAK camera (e.g. OAK-D, OAK-1)
- [pib-backend](https://github.com/pib-rocks/pib-backend) with the `motors` profile running
- ROS2 Humble with the `datatypes` package (provided by pib-backend)
- Python 3.9+

## Install

```bash
python3 -m pip install -r requirements.txt
```

ROS2 packages (`rclpy`, `datatypes`, `trajectory_msgs`) are sourced from the pib-backend ROS overlay, not pip.

## Required model files

```
models/palm_detection_sh4.blob
models/hand_landmark_full_sh4.blob
custom_models/PDPostProcessing_top2_sh1.blob
```

See [custom_models/README.md](custom_models/README.md) for building the PD post-processing blob.

## Run

Start the motors stack, then:

```bash
python demo.py
```

Optional flags:

- `-o OUTPUT` — save annotated video
- `-t [LEVEL]` — debug trace (1=app info, 2=low-level, 4=ImageManip windows, 8=save manager script)

Press `q` or Esc to quit.

## Architecture

```
demo.py
  └── HandTrackerEdge   OAK camera + DepthAI pipeline + joint-angle mapping
        └── ROS2 apply_joint_trajectory service  →  pib motor_control
  └── HandTrackerRenderer   OpenCV preview overlay
```

Fixed deployment settings (hardcoded in `HandTrackerEdge`):

- Duo mode, 2 landmark threads
- Full landmark model at 1080p, 26 FPS
- Internal frame height 640 (1152×648 preview)

## Credits

Based on [geaxgx/depthai_hand_tracker](https://github.com/geaxgx/depthai_hand_tracker) (Google Mediapipe + Luxonis DepthAI).
