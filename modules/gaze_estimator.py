import numpy as np
import logging

logger = logging.getLogger(__name__)

class GazeEstimator:
    """
    估计眼睛 gaze 向量，并将其映射至屏幕坐标。
    (简化版：仅提供基础框架和简单映射示例)
    """

    def __init__(self, screen_width_px, screen_height_px):
        """
        初始化 GazeEstimator。

        Args:
            screen_width_px (int): 目标屏幕宽度（像素）。
            screen_height_px (int): 目标屏幕高度（像素）。
        """
        self.screen_width = screen_width_px
        self.screen_height = screen_height_px
        self.calibration_model = None # 校准后得到的模型 (例如 SVR, 线性回归系数等)
        self.is_calibrated = False
        logger.info(f"GazeEstimator 初始化，目标屏幕尺寸: {screen_width_px}x{screen_height_px}")

    def load_calibration_model(self, model_data):
        """
        加载校准模型。
        (具体实现取决于所选模型)

        Args:
            model_data: 校准产生的数据或模型对象。
        """
        # 示例: 如果是简单的线性映射系数
        if isinstance(model_data, dict) and 'weights' in model_data and 'bias' in model_data:
            self.calibration_model = model_data
            self.is_calibrated = True
            logger.info("校准模型加载成功 (示例线性模型)")
            # print(f"Loaded model: {self.calibration_model}") # Debug
        else:
            logger.warning(f"尝试加载无效的校准模型数据: {type(model_data)}")
            self.is_calibrated = False


    def _extract_features(self, tracking_data: dict, face_landmarks=None, head_pose=None) -> np.ndarray | None:
        """
        从跟踪数据中提取用于注视估计的特征向量。
        (需要根据选择的估计方法具体实现)

        Args:
            tracking_data (dict): FaceMeshTracker 返回的跟踪数据。
            face_landmarks: MediaPipe 人脸关键点 (可选，用于更复杂的特征)。
            head_pose: 头部姿态信息 (可选)。

        Returns:
            np.ndarray | None: 特征向量，如果无法提取则为 None。
        """
        if not tracking_data or 'iris_centers' not in tracking_data:
            return None

        iris_left = tracking_data['iris_centers'].get('left')
        iris_right = tracking_data['iris_centers'].get('right')

        if iris_left and iris_right:
            # 示例特征：左右虹膜归一化坐标的平均值
            # 注意：这是一个极其简化的特征，实际效果可能很差
            # 更好的特征可能包括：虹膜相对眼角的位置、头部姿态补偿后的虹膜位置等
            feature_vector = np.array([
                (iris_left['x'] + iris_right['x']) / 2,
                (iris_left['y'] + iris_right['y']) / 2,
                # 可以添加更多特征，如 iris_left['z'], iris_right['z']
                # 甚至头部姿态信息 head_pose['pitch'], head_pose['yaw'] 等
            ])
            return feature_vector
        elif iris_left:
             return np.array([iris_left['x'], iris_left['y']]) # 只用左眼
        elif iris_right:
             return np.array([iris_right['x'], iris_right['y']]) # 只用右眼
        else:
            return None

    def estimate(self, tracking_data: dict, face_landmarks=None, head_pose=None) -> tuple[int | None, int | None]:
        """
        根据跟踪数据估计屏幕注视点坐标。

        Args:
            tracking_data (dict): FaceMeshTracker 返回的跟踪数据。
            face_landmarks: MediaPipe 人脸关键点 (可选)。
            head_pose: 头部姿态信息 (可选)。

        Returns:
            tuple[int | None, int | None]: 估计的屏幕坐标 (x, y)，如果无法估计则为 (None, None)。
        """
        if not self.is_calibrated or self.calibration_model is None:
            # logger.warning("系统未校准，无法进行注视点估计")
            return None, None

        features = self._extract_features(tracking_data, face_landmarks, head_pose)
        if features is None:
            # logger.debug("无法提取有效的注视特征")
            return None, None

        try:
            # --- 使用校准模型进行预测 ---
            # 示例：简单的线性映射模型 y = Wx + b
            # 假设 calibration_model = {'weights': W, 'bias': b}
            # W 的形状可能是 (2, num_features), b 的形状是 (2,)
            # features 的形状是 (num_features,)
            if isinstance(self.calibration_model, dict): # Check if it's our example model
                 weights = self.calibration_model['weights']
                 bias = self.calibration_model['bias']

                 # 确保特征维度匹配
                 if features.shape[0] != weights.shape[1]:
                     # logger.error(f"特征维度 ({features.shape[0]}) 与模型权重维度 ({weights.shape[1]}) 不匹配")
                     # Fallback: Try using only the first N features if possible? Risky.
                     # Or simply return None. Let's return None for safety.
                      logger.debug(f"特征维度 ({features.shape[0]}) 与模型权重维度 ({weights.shape[1]}) 不匹配, features: {features}")
                      return None, None


                 predicted_norm = np.dot(features, weights.T) + bias # (2,)
                 # print(f"Features: {features}, Weights: {weights.shape}, Bias: {bias.shape}, Pred Norm: {predicted_norm}") # Debug

                 # 将归一化预测值 [0, 1] 映射到屏幕像素坐标
                 screen_x = int(predicted_norm[0] * self.screen_width)
                 screen_y = int(predicted_norm[1] * self.screen_height)

                 # 限制坐标在屏幕范围内
                 screen_x = max(0, min(screen_x, self.screen_width - 1))
                 screen_y = max(0, min(screen_y, self.screen_height - 1))

                 # logger.debug(f"Estimated gaze: ({screen_x}, {screen_y})")
                 return screen_x, screen_y
            else:
                 logger.error("加载的校准模型不是预期的格式")
                 return None, None

        except Exception as e:
            logger.error(f"注视点估计时发生错误: {e}", exc_info=True)
            return None, None

    def get_gaze_direction(self, tracking_data: dict) -> str:
        """
        (极其简化的) 判断大致注视方向。

        Args:
            tracking_data (dict): FaceMeshTracker 返回的跟踪数据。

        Returns:
            str: 方向描述 ("center", "left", "right", "up", "down")
        """
        features = self._extract_features(tracking_data)
        if features is None or len(features) < 2:
            return "unknown"

        norm_x, norm_y = features[0], features[1] # Using the simplified average iris position

        # 定义中心区域阈值 (归一化坐标)
        center_thresh_x = 0.15 # e.g., 0.5 +/- 0.15 => 0.35 to 0.65
        center_thresh_y = 0.15 # e.g., 0.5 +/- 0.15 => 0.35 to 0.65

        if (0.5 - center_thresh_x) < norm_x < (0.5 + center_thresh_x) and \
           (0.5 - center_thresh_y) < norm_y < (0.5 + center_thresh_y):
            return "center"
        elif norm_x < (0.5 - center_thresh_x):
            return "left"
        elif norm_x > (0.5 + center_thresh_x):
            return "right"
        elif norm_y < (0.5 - center_thresh_y):
            # Note: Screen Y is often inverted (0 at top)
            # Assuming normalized Y is also 0 at top
            return "up"
        elif norm_y > (0.5 + center_thresh_y):
            return "down"
        else:
            # Could be on the border, classify based on dominant direction
             if abs(norm_x - 0.5) > abs(norm_y - 0.5): # More horizontal deviation
                 return "left" if norm_x < 0.5 else "right"
             else: # More vertical deviation
                 return "up" if norm_y < 0.5 else "down"
