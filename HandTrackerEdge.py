import re
import math
import marshal
import threading
from pathlib import Path
from string import Template

import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from datatypes.srv import ApplyJointTrajectory

import mediapipe_utils as mpu
from FPS import FPS

SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR / "custom_models/PDPostProcessing_top2_sh1.blob")
TEMPLATE_MANAGER_SCRIPT = str(SCRIPT_DIR / "template_manager_script_duo.py")


class HandTrackerEdge:
    """
    Edge-mode hand tracker for pib imitation.

    Runs palm detection and full landmark models on an OAK camera via DepthAI,
    maps detected hand poses to robot joint commands, and publishes them over ROS2.
    """

    def __init__(self,
                 internal_frame_height=640,
                 stats=False,
                 trace=0,
                 use_handedness_average=True,
                 single_hand_tolerance_thresh=10,
                 use_same_image=True,
                 angle_array_final=None):

        self.pd_model = PALM_DETECTION_MODEL
        print(f"Palm detection blob     : {self.pd_model}")
        self.lm_model = LANDMARK_MODEL
        print(f"Landmark blob           : {self.lm_model}")
        self.pd_score_thresh = 0.5
        self.pd_nms_thresh = 0.3
        self.lm_score_thresh = 0.5
        self.pp_model = DETECTION_POSTPROCESSING_MODEL
        print(f"PD post processing blob : {self.pp_model}")

        self.lm_nb_threads = 2
        self.resolution = (1920, 1080)
        self.internal_fps = 26
        print("Sensor resolution:", self.resolution)
        print(f"Internal camera FPS set to: {self.internal_fps}")

        self.stats = stats
        self.trace = trace
        self.use_handedness_average = use_handedness_average
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh
        self.use_same_image = use_same_image
        self.angle_array_final = angle_array_final if angle_array_final is not None else []

        self.device = dai.Device()
        self.video_fps = self.internal_fps

        rclpy.init()
        self.node = rclpy.create_node("pib_motor_client_node")
        self.apply_joint_trajectory_client = self.node.create_client(
            ApplyJointTrajectory, 'apply_joint_trajectory'
        )
        self._pending_joint_commands: dict[str, float] = {}
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        if not self.apply_joint_trajectory_client.wait_for_service(timeout_sec=15.0):
            self.node.get_logger().warn(
                "apply_joint_trajectory service not available after 15s; "
                "motor commands will be sent once the service appears"
            )

        _, self.scale_nd = mpu.find_isp_scale_params(
            internal_frame_height * self.resolution[0] / self.resolution[1],
            self.resolution,
            is_height=False,
        )
        self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.pad_h = (self.img_w - self.img_h) // 2
        self.pad_w = 0
        self.frame_size = self.img_w
        self.crop_w = 0
        print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=1, blocking=False)
        if self.trace & 4:
            self.q_pre_pd_manip_out = self.device.getOutputQueue(
                name="pre_pd_manip_out", maxSize=1, blocking=False
            )
            self.q_pre_lm_manip_out = self.device.getOutputQueue(
                name="pre_lm_manip_out", maxSize=1, blocking=False
            )

        self.fps = FPS()
        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0

    def apply_joint_trajectory(self, motor_name: str, position: float) -> None:
        self._pending_joint_commands[motor_name] = float(position)

    def _on_trajectory_done(self, future) -> None:
        try:
            result = future.result()
            if not result.successful:
                self.node.get_logger().warn("apply_joint_trajectory returned unsuccessful")
        except Exception as exc:
            self.node.get_logger().error(f"apply_joint_trajectory failed: {exc}")

    def flush_joint_trajectory(self) -> None:
        if not self._pending_joint_commands:
            return

        request = ApplyJointTrajectory.Request()
        jt = JointTrajectory()
        jt.joint_names = list(self._pending_joint_commands.keys())
        jt.points = []
        for position in self._pending_joint_commands.values():
            point = JointTrajectoryPoint()
            point.positions.append(position)
            jt.points.append(point)
        request.joint_trajectory = jt
        future = self.apply_joint_trajectory_client.call_async(request)
        future.add_done_callback(self._on_trajectory_done)
        self._pending_joint_commands.clear()

    def create_pipeline(self):
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
        self.pd_input_length = 128

        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(self.internal_fps)
        cam.setVideoSize(self.img_w, self.img_h)
        cam.setPreviewSize(self.img_w, self.img_h)

        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        cam.video.link(cam_out.input)

        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        print("Creating Palm Detection pre processing image manip...")
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length * self.pd_input_length * 3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

        if self.trace & 4:
            pre_pd_manip_out = pipeline.createXLinkOut()
            pre_pd_manip_out.setStreamName("pre_pd_manip_out")
            pre_pd_manip.out.link(pre_pd_manip_out.input)

        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        pre_pd_manip.out.link(pd_nn.input)

        print("Creating Palm Detection post processing Neural Network...")
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])

        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs['host'].link(manager_out.input)

        print("Creating Hand Landmark pre processing image manip...")
        self.lm_input_length = 224
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length * self.lm_input_length * 3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)

        if self.trace & 4:
            pre_lm_manip_out = pipeline.createXLinkOut()
            pre_lm_manip_out.setStreamName("pre_lm_manip_out")
            pre_lm_manip.out.link(pre_lm_manip_out.input)

        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)

        print(f"Creating Hand Landmark Neural Network (2 threads)...")
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(self.lm_nb_threads)
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])

        print("Pipeline created.")
        return pipeline

    def build_manager_script(self):
        with open(TEMPLATE_MANAGER_SCRIPT, 'r') as file:
            template = Template(file.read())

        code = template.substitute(
            _TRACE1="node.warn" if self.trace & 1 else "#",
            _TRACE2="node.warn" if self.trace & 2 else "#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=self.lm_score_thresh,
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w,
            _frame_size=self.frame_size,
            _crop_w=self.crop_w,
            _IF_XYZ='"""',
            _IF_USE_HANDEDNESS_AVERAGE="" if self.use_handedness_average else '"""',
            _single_hand_tolerance_thresh=self.single_hand_tolerance_thresh,
            _IF_USE_SAME_IMAGE="" if self.use_same_image else '"""',
            _IF_USE_WORLD_LANDMARKS='"""',
        )
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\n\s*\n', '\n', code)
        if self.trace & 8:
            with open("tmp_code.py", "w") as file:
                file.write(code)
        return code

    def extract_hand_data(self, res, hand_idx):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx]
        hand.rect_points = mpu.rotated_rect_to_points(
            hand.rect_x_center_a, hand.rect_y_center_a,
            hand.rect_w_a, hand.rect_h_a, hand.rotation
        )
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1, 3)
        hand.landmarks = (
            np.array(res["sqn_lms"][hand_idx]) * self.frame_size
        ).reshape(-1, 2).astype(np.int32)

        if self.pad_h > 0:
            hand.landmarks[:, 1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:, 0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        return hand

    def next_frame(self):
        self.fps.update()

        in_video = self.q_video.get()
        video_frame = in_video.getCvFrame()

        if self.trace & 4:
            pre_pd_manip = self.q_pre_pd_manip_out.tryGet()
            if pre_pd_manip:
                cv2.imshow("pre_pd_manip", pre_pd_manip.getCvFrame())
            pre_lm_manip = self.q_pre_lm_manip_out.tryGet()
            if pre_lm_manip:
                cv2.imshow("pre_lm_manip", pre_lm_manip.getCvFrame())

        res = marshal.loads(self.q_manager_out.get().getData())
        hands = []
        norm_landmarks = []
        orientation = 0

        index_hand_right = -1
        index_hand_left = -1
        shoulder_horizontal_left = 7000
        shoulder_horizontal_right = -7000

        if len(res.get("lm_score", [])) == 0:
            self.apply_joint_trajectory("shoulder_vertical_left", -7900)
            self.apply_joint_trajectory("upper_arm_left_rotation", 0)
            self.apply_joint_trajectory("shoulder_vertical_right", -7300)
            self.apply_joint_trajectory("upper_arm_right_rotation", 0)
            self.apply_joint_trajectory("elbow_left", 0)
            self.apply_joint_trajectory("elbow_right", 500)
            self.apply_joint_trajectory("lower_arm_left_rotation", 0)
            self.apply_joint_trajectory("lower_arm_right_rotation", 1100)
            self.apply_joint_trajectory("shoulder_horizontal_left", -9000)
            self.apply_joint_trajectory("shoulder_horizontal_right", -7500)

        for i in range(len(res.get("lm_score", []))):
            hand = self.extract_hand_data(res, i)
            if hand.label == "left":
                index_hand_left = i
                shoulder_horizontal_left = hand.landmarks[0][0] * 13 - 10000
                shoulder_vertical_left = 5000 - hand.landmarks[0][1] * 13 - 2000
            if hand.label == "right":
                index_hand_right = i
                shoulder_horizontal_right = hand.landmarks[0][0] * 13 - 4000
                shoulder_vertical_right = hand.landmarks[0][1] * 13 - 4000
            hands.append(hand)
            norm_landmarks.append(hand.norm_landmarks)
            orientation = hand.handedness

        angle_array = []
        if norm_landmarks:
            vec_thumb_low = []
            vec_thumb_high = []
            vec_thumb_low2 = []
            vec_thumb_high2 = []
            angle_thumb = []
            angle_thumb2 = []
            vec_idx_low = []
            vec_idx_high = []
            angle_idx = []
            vec_mid_low = []
            vec_mid_high = []
            angle_mid = []
            vec_rng_low = []
            vec_rng_high = []
            angle_rng = []
            vec_ltl_low = []
            vec_ltl_high = []
            angle_ltl = []

            for i in range(len(norm_landmarks)):
                vec_thumb_low.append(norm_landmarks[i][1][:] - norm_landmarks[i][2][:])
                vec_thumb_high.append(norm_landmarks[i][4][:] - norm_landmarks[i][3][:])
                angle_thumb.append(math.acos(np.dot(vec_thumb_high[i], vec_thumb_low[i]) / (np.linalg.norm(vec_thumb_low[i]) * np.linalg.norm(vec_thumb_high[i]))))
                vec_thumb_low2.append(norm_landmarks[i][3][:] - norm_landmarks[i][2][:])
                vec_thumb_high2.append(norm_landmarks[i][9][:] - norm_landmarks[i][0][:])
                angle_thumb2.append(math.acos(np.dot(vec_thumb_high2[i], vec_thumb_low2[i]) / (np.linalg.norm(vec_thumb_low2[i]) * np.linalg.norm(vec_thumb_high2[i]))))

                vec_idx_low.append(norm_landmarks[i][5][:] - norm_landmarks[i][6][:])
                vec_idx_high.append(norm_landmarks[i][8][:] - norm_landmarks[i][7][:])
                angle_idx.append(math.acos(np.dot(vec_idx_high[i], vec_idx_low[i]) / (np.linalg.norm(vec_idx_low[i]) * np.linalg.norm(vec_idx_high[i]))))

                vec_mid_low.append(norm_landmarks[i][9][:] - norm_landmarks[i][10][:])
                vec_mid_high.append(norm_landmarks[i][12][:] - norm_landmarks[i][11][:])
                angle_mid.append(math.acos(np.dot(vec_mid_high[i], vec_mid_low[i]) / (np.linalg.norm(vec_mid_low[i]) * np.linalg.norm(vec_mid_high[i]))))

                vec_rng_low.append(norm_landmarks[i][13][:] - norm_landmarks[i][14][:])
                vec_rng_high.append(norm_landmarks[i][16][:] - norm_landmarks[i][15][:])
                angle_rng.append(math.acos(np.dot(vec_rng_high[i], vec_rng_low[i]) / (np.linalg.norm(vec_rng_low[i]) * np.linalg.norm(vec_rng_high[i]))))

                vec_ltl_low.append(norm_landmarks[i][17][:] - norm_landmarks[i][18][:])
                vec_ltl_high.append(norm_landmarks[i][20][:] - norm_landmarks[i][19][:])
                angle_ltl.append(math.acos(np.dot(vec_ltl_high[i], vec_ltl_low[i]) / (np.linalg.norm(vec_ltl_low[i]) * np.linalg.norm(vec_ltl_high[i]))))

            if index_hand_left > -1:
                self.apply_joint_trajectory("upper_arm_right_rotation", -0.5 * shoulder_horizontal_left)
                self.apply_joint_trajectory("shoulder_vertical_right", 0.5 * shoulder_vertical_left - 1000)
                value_thumb_stretch = angle_thumb[index_hand_left] * 5000 - 9000
                self.apply_joint_trajectory("thumb_right_opposition", -2*value_thumb_stretch)
                value_idx = angle_idx[index_hand_left] * 5000 - 6000
                self.apply_joint_trajectory("index_right_stretch", -value_idx)
                value_mid = angle_mid[index_hand_left] * 5000 - 9000
                self.apply_joint_trajectory("middle_right_stretch", -value_mid)
                value_rng = angle_rng[index_hand_left] * 5000 - 9000
                self.apply_joint_trajectory("ring_right_stretch", -value_rng)
                value_ltl = angle_ltl[index_hand_left] * 5000 - 9000
                self.apply_joint_trajectory("pinky_right_stretch", -value_ltl)
                self.apply_joint_trajectory("lower_arm_right_rotation", 7400)
                self.apply_joint_trajectory("elbow_right", 3700)

            if index_hand_right > -1:
                self.apply_joint_trajectory("upper_arm_left_rotation", 0.5 * shoulder_horizontal_right)
                self.apply_joint_trajectory("shoulder_vertical_left", -0.5 * shoulder_vertical_right - 1000)
                value_thumb_stretch = angle_thumb[index_hand_right] * 5000 - 9000
                self.apply_joint_trajectory("thumb_left_opposition", -2*value_thumb_stretch)
                value_idx = angle_idx[index_hand_right] * 5000 - 9000
                self.apply_joint_trajectory("index_left_stretch", -value_idx)
                value_mid = angle_mid[index_hand_right] * 5000 - 9000
                self.apply_joint_trajectory("middle_left_stretch", -value_mid)
                value_rng = angle_rng[index_hand_right] * 5000 - 6000
                self.apply_joint_trajectory("ring_left_stretch", -value_rng)
                value_ltl = angle_ltl[index_hand_right] * 5000 - 6000
                self.apply_joint_trajectory("pinky_left_stretch", -value_ltl)
                self.apply_joint_trajectory("elbow_left", 2500)
                self.apply_joint_trajectory("lower_arm_left_rotation", 9000)

            angle_array.append(angle_thumb)
            angle_array.append(angle_thumb2)
            angle_array.append(angle_idx)
            angle_array.append(angle_mid)
            angle_array.append(angle_rng)
            angle_array.append(angle_ltl)
            angle_array = np.array(angle_array)

            for i in range(len(angle_array)):
                for j in range(len(angle_array[i])):
                    angle_array[i][j] = angle_array[i][j] * (180 / math.pi)

        self.angle_array_final = angle_array

        if len(norm_landmarks) == 2:
            if orientation < 0.5:
                self.angle_array_final[:, 0] = angle_array[:, 1]
                self.angle_array_final[:, 1] = angle_array[:, 0]

        if self.stats:
            if res["pd_inf"]:
                self.nb_frames_pd_inference += 1
            else:
                if res["nb_lm_inf"] > 0:
                    self.nb_frames_lm_inference_after_landmarks_ROI += 1
            if res["nb_lm_inf"] == 0:
                self.nb_frames_no_hand += 1
            else:
                self.nb_frames_lm_inference += 1
                self.nb_lm_inferences += res["nb_lm_inf"]
                self.nb_failed_lm_inferences += res["nb_lm_inf"] - len(hands)

        self.flush_joint_trajectory()
        return video_frame, hands, None

    def exit(self):
        self.device.close()
        if hasattr(self, "_executor"):
            self._executor.shutdown()
            self._spin_thread.join(timeout=2.0)
        if hasattr(self, "node"):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.stats:
            nb_frames = self.fps.nb_frames()
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {nb_frames})")
            print(f"# frames w/ no hand           : {self.nb_frames_no_hand} ({100 * self.nb_frames_no_hand / nb_frames:.1f}%)")
            print(f"# frames w/ palm detection    : {self.nb_frames_pd_inference} ({100 * self.nb_frames_pd_inference / nb_frames:.1f}%)")
            print(f"# frames w/ landmark inference : {self.nb_frames_lm_inference} ({100 * self.nb_frames_lm_inference / nb_frames:.1f}%) - # after palm detection: {self.nb_frames_lm_inference - self.nb_frames_lm_inference_after_landmarks_ROI} - # after landmarks ROI prediction: {self.nb_frames_lm_inference_after_landmarks_ROI}")
            if self.nb_lm_inferences:
                print(f"On frames with at least one landmark inference, average number of landmarks inferences/frame: {self.nb_lm_inferences / self.nb_frames_lm_inference:.2f}")
                print(f"# lm inferences: {self.nb_lm_inferences} - # failed lm inferences: {self.nb_failed_lm_inferences} ({100 * self.nb_failed_lm_inferences / self.nb_lm_inferences:.1f}%)")
