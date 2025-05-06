import csv
import json
import os
import time
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLogger:
    """保存注视数据、标定数据，支持导出为 JSON/CSV。"""

    def __init__(self, log_dir="data/logs", calibration_dir="data/calibration"):
        self.log_dir = log_dir
        self.calibration_dir = calibration_dir
        self.gaze_log_file = None
        self.gaze_writer = None
        self.gaze_log_buffer = [] # Buffer gaze data before writing
        self.buffer_size_limit = 100 # Write to disk every N records

        # 创建目录 (如果不存在)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.calibration_dir, exist_ok=True)

        self.gaze_log_header = [
            'timestamp', 'frame_index', # Basic info
            'screen_x', 'screen_y', # Estimated gaze
            'iris_left_x', 'iris_left_y', 'iris_left_z', # Features
            'iris_right_x', 'iris_right_y', 'iris_right_z', # Features
            'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll', # Optional features
            'blink_detected' # Optional event
        ]


    def start_gaze_log(self, filename_prefix="gaze_log"):
        """开始记录注视数据到新的 CSV 文件。"""
        self.stop_gaze_log() # Ensure previous file is closed

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.log_dir, f"{filename_prefix}_{timestamp_str}.csv")
        try:
            # Use 'a' for append if needed, but 'w' ensures a fresh file per start
            self.gaze_log_file = open(filepath, 'w', newline='', encoding='utf-8')
            self.gaze_writer = csv.DictWriter(self.gaze_log_file, fieldnames=self.gaze_log_header)
            self.gaze_writer.writeheader()
            self.gaze_log_buffer = []
            logger.info(f"开始记录注视数据到: {filepath}")
            return True
        except IOError as e:
            logger.error(f"无法打开注视数据日志文件 {filepath}: {e}", exc_info=True)
            self.gaze_log_file = None
            self.gaze_writer = None
            return False

    def log_gaze_data(self, frame_index: int, gaze_point: tuple | None, tracking_data: dict | None, head_pose: dict | None = None, blink_detected: bool = False):
        """
        记录单帧的注视相关数据。

        Args:
            frame_index (int): 当前帧序号。
            gaze_point (tuple | None): 估计的屏幕坐标 (x, y) 或 None。
            tracking_data (dict | None): FaceMeshTracker 返回的数据。
            head_pose (dict | None): 头部姿态数据 (可选)。
            blink_detected (bool): 是否检测到眨眼 (可选)。
        """
        if self.gaze_writer is None:
            # logger.warning("尝试在未开始日志记录时写入数据")
            return

        log_entry = {'timestamp': time.time(), 'frame_index': frame_index}

        log_entry['screen_x'] = gaze_point[0] if gaze_point else ''
        log_entry['screen_y'] = gaze_point[1] if gaze_point else ''

        if tracking_data and 'iris_centers' in tracking_data:
             left_iris = tracking_data['iris_centers'].get('left', {})
             right_iris = tracking_data['iris_centers'].get('right', {})
             log_entry['iris_left_x'] = left_iris.get('x', '')
             log_entry['iris_left_y'] = left_iris.get('y', '')
             log_entry['iris_left_z'] = left_iris.get('z', '')
             log_entry['iris_right_x'] = right_iris.get('x', '')
             log_entry['iris_right_y'] = right_iris.get('y', '')
             log_entry['iris_right_z'] = right_iris.get('z', '')
        else:
             log_entry.update({k: '' for k in self.gaze_log_header if 'iris_' in k})


        if head_pose:
            log_entry['head_pose_pitch'] = head_pose.get('pitch', '')
            log_entry['head_pose_yaw'] = head_pose.get('yaw', '')
            log_entry['head_pose_roll'] = head_pose.get('roll', '')
        else:
            log_entry.update({k: '' for k in self.gaze_log_header if 'head_pose_' in k})

        log_entry['blink_detected'] = '1' if blink_detected else '0'

        # Add to buffer
        self.gaze_log_buffer.append(log_entry)

        # Write buffer to disk if limit reached
        if len(self.gaze_log_buffer) >= self.buffer_size_limit:
            self._flush_gaze_buffer()


    def _flush_gaze_buffer(self):
        """Write the current buffer to the CSV file."""
        if self.gaze_writer and self.gaze_log_buffer:
            try:
                self.gaze_writer.writerows(self.gaze_log_buffer)
                self.gaze_log_buffer = []
            except Exception as e:
                 logger.error(f"写入注视数据缓冲区时出错: {e}", exc_info=True)


    def stop_gaze_log(self):
        """停止记录注视数据并关闭文件。"""
        if self.gaze_log_file:
            self._flush_gaze_buffer() # Write any remaining data
            try:
                self.gaze_log_file.close()
                logger.info("注视数据日志文件已关闭")
            except IOError as e:
                logger.error(f"关闭注视数据日志文件时出错: {e}", exc_info=True)
            finally:
                self.gaze_log_file = None
                self.gaze_writer = None


    def save_calibration_data(self, calibration_model_params: dict, filename="calibration_model.json"):
        """
        将校准模型参数保存到文件。

        Args:
            calibration_model_params (dict): Calibrator 返回的模型参数。
            filename (str): 保存的文件名。
        """
        if not calibration_model_params:
            logger.warning("没有有效的校准模型参数可保存")
            return False

        filepath = os.path.join(self.calibration_dir, filename)

        try:
            # Need to handle NumPy arrays for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist() # Convert numpy arrays to lists
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32,
                                      np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                    return {'real': obj.real, 'imag': obj.imag}
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.void)):
                    return None
                # For SVR models or other complex objects, this won't work directly.
                # You might need specific serialization logic (e.g., using pickle, joblib)
                # or just save the essential parameters like coefficients/support vectors if possible.
                # For simplicity with LinearRegression, tolist() works for coef_ and intercept_.
                # raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
                logger.warning(f"数据记录器：遇到无法直接序列化为 JSON 的类型 {type(obj)}，尝试跳过或使用默认转换。")
                # Fallback for unknown types - maybe convert to string? Risky.
                # return str(obj)
                return None # Safer to return None or skip


            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(calibration_model_params, f, indent=4, default=convert_numpy)
            logger.info(f"校准模型参数已保存到: {filepath}")
            return True
        except (IOError, TypeError) as e:
            logger.error(f"保存校准数据时发生错误: {e}", exc_info=True)
            return False

    def load_calibration_data(self, filename="calibration_model.json") -> dict | None:
        """
        从文件加载校准模型参数。

        Args:
            filename (str): 要加载的文件名。

        Returns:
            dict | None: 加载的模型参数，如果失败则为 None。
        """
        filepath = os.path.join(self.calibration_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"校准数据文件不存在: {filepath}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_params = json.load(f)

            # Convert lists back to NumPy arrays if needed by the model loader
            if 'weights' in model_params:
                model_params['weights'] = np.array(model_params['weights'])
            if 'bias' in model_params:
                model_params['bias'] = np.array(model_params['bias'])

            logger.info(f"校准模型参数已从 {filepath} 加载")
            return model_params
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"加载校准数据时发生错误: {e}", exc_info=True)
            return None

    def __del__(self):
        """确保在对象销毁时关闭文件。"""
        self.stop_gaze_log()
