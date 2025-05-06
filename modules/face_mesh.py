import cv2
import mediapipe as mp
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)

class FaceMeshTracker:
    """封装 MediaPipe 的面部/虹膜关键点提取逻辑。"""

    def __init__(self, max_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化 FaceMeshTracker。

        Args:
            max_faces (int): 检测的最大人脸数量。
            refine_landmarks (bool): 是否细化眼部和唇部的关键点（启用虹膜跟踪需要为 True）。
            min_detection_confidence (float): 人脸检测模型的最小置信度。
            min_tracking_confidence (float): 关键点跟踪模型的最小置信度。
        """
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None # Defer initialization until first use or explicit start
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 虹膜关键点索引 (左右眼，基于 MediaPipe v0.8.10+, 478 个点)
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        self.iris_indices = {
            'left': list(range(474, 478)), # 左眼虹膜 4 个点 (中心点在 473)
            'right': list(range(469, 473)) # 右眼虹膜 4 个点 (中心点在 468)
        }
        self.left_eye_center_idx = 473
        self.right_eye_center_idx = 468

        # 其他可能用到的关键点 (示例)
        self.left_eye_corner_indices = [33, 133] # 内外眼角
        self.right_eye_corner_indices = [362, 263] # 内外眼角

        self.blink_count = 0
        self._left_eye_closed = False
        self._right_eye_closed = False
        self._last_blink_time = 0

    def _initialize_mediapipe(self):
        """Lazy initialization of MediaPipe Face Mesh."""
        if self.face_mesh is None:
            try:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=self.max_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence)
                logger.info("MediaPipe FaceMesh 初始化成功")
            except Exception as e:
                logger.error(f"MediaPipe FaceMesh 初始化失败: {e}", exc_info=True)
                self.face_mesh = None # Ensure it's None if init fails


    def process_frame(self, frame: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, list | None, dict | None]:
        """
        处理单帧图像，提取面部关键点。

        Args:
            frame (cv2.typing.MatLike): 输入的 BGR 图像帧。

        Returns:
            tuple:
                - annotated_frame (cv2.typing.MatLike): 绘制了关键点的图像帧。
                - face_landmarks_list (list | None): 检测到的每个人脸的关键点列表 (归一化坐标)。
                                                    如果未检测到则为 None。
                - tracking_data (dict | None): 包含虹膜中心、眨眼次数等信息的字典。
                                                如果未检测到则为 None。
        """
        if self.face_mesh is None:
            self._initialize_mediapipe()
            if self.face_mesh is None: # Check if initialization failed
                 return frame, None, None

        # 转换颜色空间 BGR -> RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # 提高性能

        # MediaPipe 处理
        try:
            results = self.face_mesh.process(image_rgb)
        except Exception as e:
             logger.error(f"MediaPipe 处理帧时出错: {e}", exc_info=True)
             return frame, None, None # Return original frame on error

        image_rgb.flags.writeable = True
        annotated_frame = frame.copy() # 在 BGR 图像上绘制

        face_landmarks_list = None
        tracking_data = None

        if results.multi_face_landmarks:
            face_landmarks_list = []
            tracking_data = {'iris_centers': {}, 'blink_count': self.blink_count}
            image_height, image_width, _ = annotated_frame.shape

            for face_landmarks in results.multi_face_landmarks:
                face_landmarks_list.append(face_landmarks) # 存储原始 landmark 对象

                # --- 绘制面部网格 (可选) ---
                # self.mp_drawing.draw_landmarks(
                #     image=annotated_frame,
                #     landmark_list=face_landmarks,
                #     connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)) # 绿色网格

                # --- 绘制眼睛轮廓和虹膜 (可选) ---
                # self.mp_drawing.draw_landmarks(
                #     image=annotated_frame,
                #     landmark_list=face_landmarks,
                #     connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=1)) # 左眼蓝色
                # self.mp_drawing.draw_landmarks(
                #     image=annotated_frame,
                #     landmark_list=face_landmarks,
                #     connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)) # 右眼红色
                # self.mp_drawing.draw_landmarks(
                #     image=annotated_frame,
                #     landmark_list=face_landmarks,
                #     connections=self.mp_face_mesh.FACEMESH_LEFT_IRIS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,0), thickness=1)) # 左虹膜青色
                # self.mp_drawing.draw_landmarks(
                #     image=annotated_frame,
                #     landmark_list=face_landmarks,
                #     connections=self.mp_face_mesh.FACEMESH_RIGHT_IRIS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)) # 右虹膜黄色

                # --- 提取并绘制虹膜中心点 ---
                landmarks = face_landmarks.landmark
                # 左眼虹膜中心
                left_iris_center_norm = landmarks[self.left_eye_center_idx]
                lx, ly = int(left_iris_center_norm.x * image_width), int(left_iris_center_norm.y * image_height)
                cv2.circle(annotated_frame, (lx, ly), radius=2, color=(0, 255, 0), thickness=-1) # 绿色点
                tracking_data['iris_centers']['left'] = {'x': left_iris_center_norm.x, 'y': left_iris_center_norm.y, 'z': left_iris_center_norm.z}

                # 右眼虹膜中心
                right_iris_center_norm = landmarks[self.right_eye_center_idx]
                rx, ry = int(right_iris_center_norm.x * image_width), int(right_iris_center_norm.y * image_height)
                cv2.circle(annotated_frame, (rx, ry), radius=2, color=(0, 0, 255), thickness=-1) # 红色点
                tracking_data['iris_centers']['right'] = {'x': right_iris_center_norm.x, 'y': right_iris_center_norm.y, 'z': right_iris_center_norm.z}

                # --- 简单的眨眼检测 ---
                self._detect_blink(landmarks, image_width, image_height)
                tracking_data['blink_count'] = self.blink_count


                # --- 可以在这里添加头部姿态估计的调用 (如果需要的话) ---
                # head_pose = self.estimate_head_pose(landmarks, image_width, image_height)
                # tracking_data['head_pose'] = head_pose
                # annotated_frame = self.draw_pose_axes(annotated_frame, head_pose)


        return annotated_frame, face_landmarks_list, tracking_data

    def _calculate_ear(self, eye_landmarks_indices, landmarks, image_width, image_height):
        """计算眼睛纵横比 (Eye Aspect Ratio - EAR)。"""
        # 提取垂直方向关键点坐标 (像素)
        p2 = landmarks[eye_landmarks_indices[1]] # 上眼睑中间点
        p6 = landmarks[eye_landmarks_indices[5]] # 下眼睑中间点
        p3 = landmarks[eye_landmarks_indices[2]] # 上眼睑侧点
        p5 = landmarks[eye_landmarks_indices[4]] # 下眼睑侧点

        # 提取水平方向关键点坐标 (像素)
        p1 = landmarks[eye_landmarks_indices[0]] # 眼角点
        p4 = landmarks[eye_landmarks_indices[3]] # 眼角点

        # 转换为像素坐标
        p1_px = np.array([p1.x * image_width, p1.y * image_height])
        p2_px = np.array([p2.x * image_width, p2.y * image_height])
        p3_px = np.array([p3.x * image_width, p3.y * image_height])
        p4_px = np.array([p4.x * image_width, p4.y * image_height])
        p5_px = np.array([p5.x * image_width, p5.y * image_height])
        p6_px = np.array([p6.x * image_width, p6.y * image_height])


        # 计算垂直距离
        ver_dist1 = np.linalg.norm(p2_px - p6_px)
        ver_dist2 = np.linalg.norm(p3_px - p5_px)

        # 计算水平距离
        hor_dist = np.linalg.norm(p1_px - p4_px)

        # 计算 EAR
        if hor_dist == 0: return 0.0 # 避免除零错误
        ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
        return ear

    def _detect_blink(self, landmarks, image_width, image_height, ear_threshold=0.2, blink_debounce_time=0.2):
        """基于 EAR 检测眨眼次数。"""
        # MediaPipe 脸部关键点索引 (适用于468/478点模型)
        # 参考: https://github.com/google/mediapipe/issues/1615#issuecomment-757987003
        # 或 https://raw.githubusercontent.com/google/mediapipe/a908d668c791781a844d53899141097b61916711/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        left_eye_indices = [
            33,  # Left eye corner
            160, # Top eyelid center
            158, # Top eyelid side
            133, # Right eye corner (of left eye)
            153, # Bottom eyelid side
            144  # Bottom eyelid center
        ]
        right_eye_indices = [
            362, # Left eye corner (of right eye)
            385, # Top eyelid side
            387, # Top eyelid center
            263, # Right eye corner
            373, # Bottom eyelid center
            380  # Bottom eyelid side
        ]

        left_ear = self._calculate_ear(left_eye_indices, landmarks, image_width, image_height)
        right_ear = self._calculate_ear(right_eye_indices, landmarks, image_width, image_height)

        current_time = time.time()

        left_closed_now = left_ear < ear_threshold
        right_closed_now = right_ear < ear_threshold

        # 如果之前是睁开，现在是闭合，则标记为闭合开始
        if not self._left_eye_closed and left_closed_now:
            self._left_eye_closed = True
        if not self._right_eye_closed and right_closed_now:
            self._right_eye_closed = True

        # 如果之前是闭合，现在是睁开，并且距离上次眨眼超过抖动时间，则计为一次眨眼
        blink_detected = False
        if self._left_eye_closed and not left_closed_now:
            self._left_eye_closed = False
            blink_detected = True
        if self._right_eye_closed and not right_closed_now:
            self._right_eye_closed = False
            blink_detected = True # Any eye opening counts

        if blink_detected and (current_time - self._last_blink_time > blink_debounce_time):
             self.blink_count += 1
             self._last_blink_time = current_time
             logger.debug(f"Blink detected! Count: {self.blink_count}")


    def close(self):
        """释放 MediaPipe 资源。"""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
            logger.info("MediaPipe FaceMesh 资源已释放")

    def __del__(self):
        self.close()
