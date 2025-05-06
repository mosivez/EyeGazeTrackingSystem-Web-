import numpy as np
import logging

logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    """比较 gaze 预测点与 ground truth 的误差，输出可视化所需数据。"""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = [] # 校准时使用的目标点 [(x1,y1), (x2,y2), ...]
        self.validation_predictions = {} # 存储验证时的预测 { (tx,ty): [(px1,py1), (px2,py2),..], ... }
        self.errors = [] # 存储每次验证的误差 [e1, e2, ...] (例如像素距离)

    def set_calibration_points(self, points: list[tuple[int, int]]):
        """设置用于验证的基准点 (通常是校准点)。"""
        self.calibration_points = points
        self.validation_predictions = {pt: [] for pt in points}
        self.errors = []
        logger.info(f"误差分析器设置了 {len(points)} 个基准点")

    def record_prediction(self, target_point: tuple[int, int], predicted_point: tuple[int | None, int | None]):
        """
        记录一次验证预测。

        Args:
            target_point (tuple[int, int]): 用户被要求注视的目标点。
            predicted_point (tuple[int | None, int | None]): 系统估计的注视点。
        """
        if predicted_point[0] is None or predicted_point[1] is None:
            logger.debug("无法记录预测：无效的预测点")
            return

        pred_x, pred_y = predicted_point
        target_x, target_y = target_point

        if target_point in self.validation_predictions:
            self.validation_predictions[target_point].append((pred_x, pred_y))

            # 计算欧氏距离误差
            error = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
            self.errors.append(error)
            # logger.debug(f"记录误差: 目标={target_point}, 预测={predicted_point}, 误差={error:.2f}px")
        else:
            logger.warning(f"尝试记录未知的目标点 {target_point} 的预测")


    def calculate_statistics(self) -> dict:
        """计算误差统计信息。"""
        if not self.errors:
            return {"mean_error": None, "std_dev": None, "count": 0}

        mean_error = np.mean(self.errors)
        std_dev = np.std(self.errors)
        count = len(self.errors)

        logger.info(f"误差统计: 平均={mean_error:.2f}px, 标准差={std_dev:.2f}px, 样本数={count}")
        return {"mean_error": round(mean_error, 2), "std_dev": round(std_dev, 2), "count": count}

    def get_error_visualization_data(self, type='heatmap') -> dict:
        """
        生成用于前端可视化的数据。

        Args:
            type (str): 可视化类型 ('heatmap', 'vector').

        Returns:
            dict: 包含可视化所需数据。
                  例如 heatmap: {'max': max_error, 'data': [{'x': px, 'y': py, 'value': error}, ...]}
                  例如 vector: [{'target': [tx,ty], 'predictions': [[px1,py1], [px2,py2],...]}, ...]
        """
        if type == 'heatmap':
            # 生成热图数据点 (目标点位置，误差大小作为权重)
            heatmap_data = []
            max_error = 0
            for target, preds in self.validation_predictions.items():
                 if not preds: continue
                 target_x, target_y = target
                 # 计算该点的平均误差或最后一次误差？用平均误差
                 point_errors = [np.sqrt((px - target_x)**2 + (py - target_y)**2) for px, py in preds]
                 avg_error = np.mean(point_errors) if point_errors else 0
                 heatmap_data.append({'x': target_x, 'y': target_y, 'value': round(avg_error, 2)})
                 if avg_error > max_error: max_error = avg_error

            return {'max': round(max_error, 2), 'data': heatmap_data}

        elif type == 'vector':
            # 生成矢量图数据 (目标点 -> 预测点集合)
            vector_data = []
            for target, preds in self.validation_predictions.items():
                 if not preds: continue
                 vector_data.append({'target': list(target), 'predictions': preds})
            return vector_data

        else:
            logger.warning(f"不支持的误差可视化类型: {type}")
            return {}

    def reset(self):
        """重置误差分析数据。"""
        self.validation_predictions = {pt: [] for pt in self.calibration_points}
        self.errors = []
        logger.info("误差分析器已重置")
