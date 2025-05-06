import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoCaptureManager:
    """管理摄像头采集、图像帧获取、释放资源。"""

    def __init__(self, source=0):
        """
        初始化 VideoCaptureManager。

        Args:
            source (int or str): 摄像头索引或视频文件路径。
        """
        self.source = source
        self.cap = None
        self.is_running = False
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self._frame_count = 0
        self._start_time = 0

    def start(self) -> bool:
        """启动摄像头捕获。"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"无法打开视频源: {self.source}")
                self.cap = None
                return False

            self.is_running = True
            # 获取并存储分辨率
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"摄像头启动成功，分辨率: {self.frame_width}x{self.frame_height}")
            self._start_time = time.time()
            self._frame_count = 0
            return True
        except Exception as e:
            logger.error(f"启动摄像头时发生错误: {e}", exc_info=True)
            self.release()
            return False

    def get_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """
        获取一帧图像。

        Returns:
            tuple[bool, cv2.typing.MatLike | None]: (是否成功获取帧, 图像帧)
        """
        if not self.is_running or self.cap is None:
            # logger.warning("尝试在未运行状态下获取帧")
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("无法读取到帧")
                # self.stop() # Decide if failure to read means stop
                return False, None

            # 计算 FPS
            self._frame_count += 1
            elapsed_time = time.time() - self._start_time
            if elapsed_time >= 1.0:
                self.fps = self._frame_count / elapsed_time
                self._start_time = time.time()
                self._frame_count = 0

            return True, frame
        except Exception as e:
            logger.error(f"获取帧时发生错误: {e}", exc_info=True)
            return False, None

    def stop(self):
        """停止摄像头捕获。"""
        self.is_running = False
        logger.info("摄像头停止指令已接收")

    def release(self):
        """释放摄像头资源。"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("摄像头资源已释放")

    def get_info(self) -> dict:
        """获取摄像头信息（分辨率、FPS）。"""
        return {
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": round(self.fps, 2)
        }

    def __del__(self):
        """确保在对象销毁时释放资源。"""
        self.release()
