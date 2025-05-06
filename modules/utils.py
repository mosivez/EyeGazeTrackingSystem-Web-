import cv2
import numpy as np

def draw_gaze_point(frame, point, color=(0, 255, 0), radius=10, thickness=-1):
    """在图像上绘制注视点。"""
    if point and point[0] is not None and point[1] is not None:
        try:
            center = (int(point[0]), int(point[1]))
            cv2.circle(frame, center, radius, color, thickness)
            # Add a crosshair for better visibility
            cv2.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (255, 255, 255), 1)
            cv2.line(frame, (center[0], center[1] - radius), (center[0], center[1] + radius), (255, 255, 255), 1)
        except Exception as e:
            # logger is not defined here, maybe pass it or ignore
            print(f"Error drawing gaze point {point}: {e}")
    return frame

def draw_calibration_point(frame, point, color=(0, 0, 255), radius=15, thickness=2):
    """在图像上绘制校准目标点。"""
    if point:
        try:
            center = (int(point[0]), int(point[1]))
            cv2.circle(frame, center, radius, color, thickness)
            # Add inner circle
            cv2.circle(frame, center, radius // 2, (255, 255, 255), thickness=-1)
        except Exception as e:
             print(f"Error drawing calibration point {point}: {e}")
    return frame

def add_text_to_frame(frame, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, color=(255, 255, 255), thickness=2):
    """在图像上添加文本。"""
    try:
        cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)
    except Exception as e:
         print(f"Error adding text '{text}': {e}")
    return frame

# --- 头部姿态估计 (简化示例 - 需要 3D 模型点) ---
# 这部分比较复杂，依赖于精确的 3D 面部模型点和相机内参
# 如果你的原始项目中有实现，可以迁移过来，否则需要额外实现

# def estimate_head_pose(landmarks, image_width, image_height, camera_matrix=None, dist_coeffs=None):
#     """(简化示例) 估计头部姿态。"""
#     if camera_matrix is None:
#         # 估算相机内参 (非常粗略)
#         focal_length = image_width
#         center = (image_width / 2, image_height / 2)
#         camera_matrix = np.array(
#             [[focal_length, 0, center[0]],
#              [0, focal_length, center[1]],
#              [0, 0, 1]], dtype="double"
#         )
#     if dist_coeffs is None:
#         dist_coeffs = np.zeros((4, 1)) # 假设无畸变

#     # 需要选择一组 2D 图像点和对应的 3D 模型点
#     # 例如：鼻尖(1), 下巴(152), 左眼角(33), 右眼角(263), 左嘴角(61), 右嘴角(291)
#     image_points_indices = [1, 152, 33, 263, 61, 291]
#     # 这些 3D 点需要来自标准面部模型 (单位任意，但需一致)
#     # !!! 下面的 3D 点是示例值，你需要查找 MediaPipe 使用的或自己定义一个 !!!
#     model_points = np.array([
#         (0.0, 0.0, 0.0),             # Nose tip
#         (0.0, -330.0, -65.0),        # Chin
#         (-225.0, 170.0, -135.0),     # Left eye left corner
#         (225.0, 170.0, -135.0),      # Right eye right corner
#         (-150.0, -150.0, -125.0),    # Left Mouth corner
#         (150.0, -150.0, -125.0)      # Right mouth corner
#     ])

#     image_points = np.array([
#         (landmarks[i].x * image_width, landmarks[i].y * image_height) for i in image_points_indices
#     ], dtype="double")

#     try:
#         (success, rotation_vector, translation_vector) = cv2.solvePnP(
#             model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE # or SOLVEPNP_EPNP
#         )

#         if success:
#             # 将旋转向量转换为旋转矩阵
#             rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#             # 从旋转矩阵计算欧拉角 (需要注意旋转顺序和奇异性)
#             # ... (这部分计算比较复杂，可以查找相关函数实现) ...
#             # 简化：直接返回旋转向量和平移向量
#             # pitch, yaw, roll = rotationMatrixToEulerAngles(rotation_matrix)
#             return {
#                 "rotation_vector": rotation_vector.flatten().tolist(),
#                 "translation_vector": translation_vector.flatten().tolist(),
#                 # "pitch": pitch, "yaw": yaw, "roll": roll # 如果计算了欧拉角
#             }
#         else:
#             return None
#     except Exception as e:
#         print(f"Error estimating head pose: {e}")
#         return None

# def draw_pose_axes(frame, head_pose_data, camera_matrix, dist_coeffs, axis_length=50):
#      """在图像上绘制头部姿态坐标轴。"""
#      if head_pose_data is None: return frame

#      rotation_vector = np.array(head_pose_data["rotation_vector"]).reshape(-1, 1)
#      translation_vector = np.array(head_pose_data["translation_vector"]).reshape(-1, 1)

#      # 定义 3D 坐标轴的端点
#      axis_points_3d = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
#      # 将 3D 点投影到 2D 图像平面
#      axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

#      # 转换坐标为整数
#      origin = tuple(np.int32(axis_points_2d[0].ravel()))
#      x_axis_end = tuple(np.int32(axis_points_2d[1].ravel()))
#      y_axis_end = tuple(np.int32(axis_points_2d[2].ravel()))
#      z_axis_end = tuple(np.int32(axis_points_2d[3].ravel()))

#      # 绘制坐标轴
#      cv2.line(frame, origin, x_axis_end, (0, 0, 255), 3)  # X 轴: 红色
#      cv2.line(frame, origin, y_axis_end, (0, 255, 0), 3)  # Y 轴: 绿色
#      cv2.line(frame, origin, z_axis_end, (255, 0, 0), 3)  # Z 轴: 蓝色

#      return frame
