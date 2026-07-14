import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MODEL_PATH = str(Path(__file__).resolve().parent / "model" / "hand.task")
# MODEL_PATH = './model/hand.task'


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0], 
    ]
)

class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        # self.hand_detector = mp.solutions.hands.Hands(
        #     static_image_mode=False,
        #     max_num_hands=1,
        #     min_detection_confidence=min_detection_confidence,
        #     min_tracking_confidence=min_tracking_confidence,
        # )
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
                                         running_mode=mp.tasks.vision.RunningMode.VIDEO)
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        # self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]
        self.detected_hand_type = hand_type if not selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d: list, style="white"
    ):
        mp_drawing.draw_landmarks(
            image, 
            keypoint_2d,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        return image

    def detect(self, rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

        if not results.hand_landmarks:
            return 0, None, None, None

        desired_hand_num = -1
        for i in range(len(results.hand_landmarks)):
            label = results.handedness[i][0].category_name
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None

        keypoint_3d = results.hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.hand_landmarks[desired_hand_num]
        num_box = len(results.hand_landmarks)

        # Parse 3d keypoint from MediaPipe hand detector
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano


        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: list,
    ) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d[i].x
            keypoint[i][1] = keypoint_3d[i].y
            keypoint[i][2] = keypoint_3d[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: list, img_size
    ) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d[i].x
            keypoint[i][1] = keypoint_2d[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
