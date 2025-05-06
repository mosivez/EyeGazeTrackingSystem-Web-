import numpy as np
import time
import logging
from sklearn.linear_model import LinearRegression # Example model
# from sklearn.svm import SVR # Another option
# from sklearn.preprocessing import PolynomialFeatures # For polynomial regression
# from sklearn.pipeline import make_pipeline # To combine steps

logger = logging.getLogger(__name__)

class Calibrator:
    """控制用户校准流程，记录样本对，训练映射模型。"""

    def __init__(self, screen_width, screen_height, points_to_calibrate=None):
        """
        初始化 Calibrator。

        Args:
            screen_width (int): 屏幕宽度。
            screen_height (int): 屏幕高度。
            points_to_calibrate (list[tuple[int, int]] | None): 需要校准的屏幕点坐标列表。
                                                               如果为 None，使用默认点。
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        if points_to_calibrate is None:
            # 默认 9 点校准 (边角 + 中心 + 边缘中点)
            margin_x = int(screen_width * 0.1)
            margin_y = int(screen_height * 0.1)
            center_x = screen_width // 2
            center_y = screen_height // 2
            self.points_to_calibrate = [
                (margin_x, margin_y), (center_x, margin_y), (screen_width - margin_x, margin_y),
                (margin_x, center_y), (center_x, center_y), (screen_width - margin_x, center_y),
                (margin_x, screen_height - margin_y), (center_x, screen_height - margin_y), (screen_width - margin_x, screen_height - margin_y)
            ]
        else:
            self.points_to_calibrate = points_to_calibrate

        self.calibration_data = [] # 存储 (feature_vector, screen_coords) 对
        self.is_calibrating = False
        self.current_point_index = -1
        self.last_sample_time = 0
        self.min_sample_interval = 0.1 # 秒，防止过快采样同一点

        # 选择一个机器学习模型 (示例: 线性回归)
        # 输入: 特征向量 (e.g., 归一化虹膜位置 x, y, z?)
        # 输出: 归一化屏幕坐标 (norm_x, norm_y)
        self.model = LinearRegression()
        # self.model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()) # 多项式回归
        # self.model_x = SVR(kernel='rbf', C=1.0, epsilon=0.01) # SVR for X
        # self.model_y = SVR(kernel='rbf', C=1.0, epsilon=0.01) # SVR for Y


    def start_calibration(self):
        """开始或重新开始校准流程。"""
        self.calibration_data = []
        self.current_point_index = 0
        self.is_calibrating = True
        logger.info(f"校准开始，共 {len(self.points_to_calibrate)} 个点。请注视第一个点。")
        return self.get_current_point()

    def get_current_point(self) -> tuple[int, int] | None:
        """获取当前需要注视的校准点坐标。"""
        if self.is_calibrating and 0 <= self.current_point_index < len(self.points_to_calibrate):
            return self.points_to_calibrate[self.current_point_index]
        return None

    def add_calibration_point(self, feature_vector: np.ndarray):
        """
        记录当前注视点的特征向量。

        Args:
            feature_vector (np.ndarray): GazeEstimator 提取的特征向量。

        Returns:
            bool: 是否成功添加数据点。
        """
        current_point = self.get_current_point()
        current_time = time.time()

        if current_point is None or feature_vector is None:
            logger.warning("无法添加校准点：无效状态或无效特征")
            return False

        # 简单防抖：确保与上次采样间隔足够
        if current_time - self.last_sample_time < self.min_sample_interval:
             # logger.debug("采样间隔过短，忽略此次数据")
             return False


        # 将屏幕坐标归一化到 [0, 1]
        norm_x = current_point[0] / self.screen_width
        norm_y = current_point[1] / self.screen_height
        screen_coords_normalized = np.array([norm_x, norm_y])

        self.calibration_data.append((feature_vector, screen_coords_normalized))
        self.last_sample_time = current_time
        logger.info(f"记录点 {self.current_point_index + 1}/{len(self.points_to_calibrate)} 的数据。特征: {feature_vector.round(3)}, 目标: {screen_coords_normalized.round(3)}")
        # print(f"Data added for point {self.current_point_index}: Feature={feature_vector}, Target={screen_coords_normalized}") # Debug
        return True

    def next_point(self) -> tuple[int, int] | None:
        """移动到下一个校准点。"""
        if not self.is_calibrating:
            return None

        self.current_point_index += 1
        if self.current_point_index >= len(self.points_to_calibrate):
            logger.info("所有校准点已完成数据采集。")
            # self.finish_calibration() # Optionally finish immediately
            return None # Indicate completion
        else:
            logger.info(f"请注视下一个点 ({self.current_point_index + 1}/{len(self.points_to_calibrate)})。")
            self.last_sample_time = 0 # Reset debounce timer for the new point
            return self.get_current_point()

    def finish_calibration(self) -> dict | None:
        """
        结束校准流程，并训练模型。

        Returns:
            dict | None: 训练好的模型参数 (或 None 如果失败)。
                         示例: {'weights': W, 'bias': b}
        """
        if not self.is_calibrating:
            logger.warning("尝试在非校准状态下结束校准")
            return None
        if len(self.calibration_data) < len(self.points_to_calibrate) or len(self.calibration_data) < 3: # Need enough points
            logger.error(f"校准数据不足 ({len(self.calibration_data)}/{len(self.points_to_calibrate)})，无法训练模型。")
            self.is_calibrating = False
            self.current_point_index = -1
            return None

        self.is_calibrating = False
        self.current_point_index = -1
        logger.info("校准流程结束，开始训练模型...")

        try:
            # 准备训练数据
            X = np.array([data[0] for data in self.calibration_data]) # 特征
            Y = np.array([data[1] for data in self.calibration_data]) # 归一化屏幕坐标 (N, 2)

            # print(f"Training Data X shape: {X.shape}, Y shape: {Y.shape}") # Debug
            # print(f"Sample X: {X[0] if len(X)>0 else 'N/A'}")
            # print(f"Sample Y: {Y[0] if len(Y)>0 else 'N/A'}")


            # --- 训练模型 (以 LinearRegression 为例) ---
            self.model.fit(X, Y)

            # 获取模型参数 (对于线性回归)
            weights = self.model.coef_ # (2, num_features)
            bias = self.model.intercept_ # (2,)

            model_params = {'weights': weights, 'bias': bias}
            logger.info("校准模型训练完成。")
            # print(f"Trained model: Weights={weights.round(3)}, Bias={bias.round(3)}") # Debug

            # --- 如果使用 SVR ---
            # self.model_x.fit(X, Y[:, 0]) # Train for X
            # self.model_y.fit(X, Y[:, 1]) # Train for Y
            # model_params = {'model_x': self.model_x, 'model_y': self.model_y} # Store the trained models
            # logger.info("SVR 校准模型训练完成。")


            return model_params

        except Exception as e:
            logger.error(f"训练校准模型时发生错误: {e}", exc_info=True)
            return None

    def get_state(self) -> dict:
        """获取当前校准状态。"""
        return {
            "is_calibrating": self.is_calibrating,
            "current_point_index": self.current_point_index,
            "total_points": len(self.points_to_calibrate),
            "current_target_point": self.get_current_point(),
            "data_collected_count": len(self.calibration_data)
        }
