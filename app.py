import cv2
import time
import threading
import logging
from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import mediapipe as mp
# 从模块导入类
from modules.video_capture import VideoCaptureManager
from modules.face_mesh import FaceMeshTracker
from modules.gaze_estimator import GazeEstimator
from modules.calibrator import Calibrator
from modules.error_analyzer import ErrorAnalyzer
from modules.data_logger import DataLogger
import modules.utils as utils # 导入辅助函数

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flask 应用初始化 ---
app = Flask(__name__)

# --- 全局变量和状态管理 ---
# 注意：在生产环境中，对于多用户或复杂状态，需要更健壮的状态管理机制
# 这里为了简化，使用全局变量，但在 Web 应用中需要特别小心线程安全问题
camera_source = 0 # 0 for default camera
video_manager = VideoCaptureManager(source=camera_source)
face_tracker = FaceMeshTracker(refine_landmarks=True) # 启用虹膜跟踪

# 屏幕尺寸需要根据你的前端显示区域或实际屏幕来设定
# 临时设定，后续可能需要从前端获取或配置
SCREEN_WIDTH_PX = 1920 # Example: Full HD width
SCREEN_HEIGHT_PX = 1080 # Example: Full HD height

gaze_estimator = GazeEstimator(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
calibrator = Calibrator(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
error_analyzer = ErrorAnalyzer(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
data_logger = DataLogger()

# 系统状态
system_running = False
show_mesh = True # 控制是否在视频流中绘制面部网格
last_frame = None
last_processed_data = {} # 存储最新的处理结果供 status API 获取
frame_lock = threading.Lock() # 保护对 last_frame 的访问
data_lock = threading.Lock() # 保护对 last_processed_data 的访问
processing_thread = None

# --- Flask 路由定义 ---

@app.route('/')
def index():
    """提供主 HTML 页面。"""
    return render_template('index.html')

def generate_frames():
    """生成视频帧流，用于在网页上显示。"""
    global last_frame, system_running
    logger.info("视频流生成器启动")
    while True:
        with frame_lock:
            if not system_running or last_frame is None:
                 # 如果系统停止或没有帧，可以发送一个占位符图像或等待
                 # 这里简单地跳过，等待下一帧
                 time.sleep(0.05) # 避免忙等
                 continue

            # 编码帧为 JPEG
            try:
                flag, encoded_image = cv2.imencode('.jpg', last_frame)
                if not flag:
                    logger.warning("帧编码失败")
                    continue
                frame_bytes = encoded_image.tobytes()
            except Exception as e:
                 logger.error(f"帧编码或转换字节时出错: {e}")
                 continue

        # 使用 multipart/x-mixed-replace 格式发送帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # 控制发送速率，减轻浏览器负担 (约 30 FPS)

    logger.info("视频流生成器停止")


@app.route('/video_feed')
def video_feed():
    """视频流路由。"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def process_video():
    """在后台线程中处理视频帧。"""
    global last_frame, system_running, last_processed_data
    frame_index = 0
    logger.info("视频处理线程启动")

    if not video_manager.start():
         logger.error("无法启动视频管理器，处理线程退出。")
         system_running = False # Make sure state reflects failure
         return

    # 尝试加载之前的校准模型
    loaded_model = data_logger.load_calibration_data()
    if loaded_model:
        gaze_estimator.load_calibration_model(loaded_model)
        logger.info("成功加载之前的校准模型")
    else:
        logger.warning("未找到或加载校准模型失败，需要进行校准。")


    while system_running:
        success, frame = video_manager.get_frame()
        if not success or frame is None:
            # logger.debug("无法获取帧，跳过处理")
            time.sleep(0.01) # 短暂暂停
            continue

        frame_index += 1
        start_time = time.time()

        # 1. 面部和虹膜检测
        annotated_frame, face_landmarks_list, tracking_data = face_tracker.process_frame(frame)

        # 初始化当前帧的处理数据
        current_data = {
            'timestamp': time.time(),
            'frame_index': frame_index,
            'fps': video_manager.get_info()['fps'],
            'resolution': f"{video_manager.get_info()['width']}x{video_manager.get_info()['height']}",
            'face_detected': False,
            'gaze_point': None,
            'gaze_direction': 'unknown',
            'calibration_state': calibrator.get_state(),
            'is_calibrated': gaze_estimator.is_calibrated,
            'blink_count': face_tracker.blink_count,
            'error_stats': None # Placeholder for error info
        }

        estimated_gaze = None # 重置估计值

        if face_landmarks_list and tracking_data:
            current_data['face_detected'] = True
            # (可选) 提取头部姿态 - 如果实现了 estimate_head_pose
            # head_pose = utils.estimate_head_pose(face_landmarks_list[0].landmark, video_manager.frame_width, video_manager.frame_height)
            # if head_pose:
            #     annotated_frame = utils.draw_pose_axes(annotated_frame, head_pose, ...) # 需要相机参数
            #     current_data['head_pose'] = head_pose # 添加到数据
            head_pose = None # 暂时禁用

            # 2. 如果在校准模式
            if calibrator.is_calibrating:
                target_point = calibrator.get_current_point()
                if target_point:
                    # 在帧上绘制校准目标点
                    annotated_frame = utils.draw_calibration_point(annotated_frame, target_point)
                    # 提取用于校准的特征
                    features = gaze_estimator._extract_features(tracking_data, face_landmarks_list[0], head_pose)
                    if features is not None:
                         # 自动或手动触发添加数据点？这里先假设前端会发指令
                         # calibrator.add_calibration_point(features)
                         pass # Wait for /api/add_calibration_point command
                    else:
                         current_data['status_message'] = "校准中：无法提取特征"
                else:
                     current_data['status_message'] = "校准点获取错误"


            # 3. 如果已校准，进行注视点估计
            elif gaze_estimator.is_calibrated:
                estimated_gaze = gaze_estimator.estimate(tracking_data, face_landmarks_list[0], head_pose)
                if estimated_gaze:
                    current_data['gaze_point'] = estimated_gaze
                    # 在帧上绘制注视点
                    annotated_frame = utils.draw_gaze_point(annotated_frame, estimated_gaze)
                    # (可选) 记录误差分析数据 - 需要知道目标点
                    # if error_analyzer.is_validating: # Need a validation mode
                    #    target = error_analyzer.get_current_target()
                    #    error_analyzer.record_prediction(target, estimated_gaze)

                # 获取大致方向
                current_data['gaze_direction'] = gaze_estimator.get_gaze_direction(tracking_data)

            else:
                # 未校准，提示用户
                utils.add_text_to_frame(annotated_frame, "Please Calibrate", (50, 50), color=(0, 0, 255))
                current_data['status_message'] = "需要校准"

            # 4. (可选) 绘制面部网格等
            if show_mesh and face_landmarks_list:
                 mp.solutions.drawing_utils.draw_landmarks(
                      image=annotated_frame,
                      landmark_list=face_landmarks_list[0],
                      connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,128,0), thickness=1))


        # 5. 添加状态文本信息到帧
        status_text_y = 30
        status_texts = [
            f"FPS: {current_data['fps']:.1f}",
            f"Res: {current_data['resolution']}",
            f"Blinks: {current_data['blink_count']}",
            f"Status: {'Running' if system_running else 'Stopped'}",
            f"Calibrated: {'Yes' if current_data['is_calibrated'] else 'No'}",
            f"Gaze Dir: {current_data['gaze_direction']}",
        ]
        if calibrator.is_calibrating:
            cal_state = current_data['calibration_state']
            status_texts.append(f"Calibrating: Point {cal_state['current_point_index']+1}/{cal_state['total_points']}")
        if current_data.get('status_message'):
             status_texts.append(f"Msg: {current_data['status_message']}")


        for text in status_texts:
            utils.add_text_to_frame(annotated_frame, text, (10, status_text_y))
            status_text_y += 25


        # 6. 更新全局帧和数据
        with frame_lock:
            last_frame = annotated_frame.copy()
        with data_lock:
            last_processed_data = current_data.copy() # Update shared data

        # 7. 记录日志 (如果已启动)
        # data_logger.log_gaze_data(frame_index, estimated_gaze, tracking_data, head_pose, ...)


        # 控制处理速率 (如果需要)
        # proc_time = time.time() - start_time
        # sleep_time = max(0, (1.0 / TARGET_FPS) - proc_time)
        # time.sleep(sleep_time)

    # 线程结束，清理资源
    video_manager.release()
    face_tracker.close()
    data_logger.stop_gaze_log() #确保日志文件关闭
    logger.info("视频处理线程已停止并清理资源")


# --- API Endpoints ---

@app.route('/api/start', methods=['POST'])
def start_system():
    """启动视频处理线程。"""
    global system_running, processing_thread
    if not system_running:
        system_running = True
        # 启动后台处理线程
        if processing_thread is None or not processing_thread.is_alive():
             processing_thread = threading.Thread(target=process_video, daemon=True)
             processing_thread.start()
             logger.info("系统启动指令已发送")
             data_logger.start_gaze_log() # 开始记录日志
             return jsonify({"status": "success", "message": "System started"})
        else:
             logger.warning("系统已在运行中")
             return jsonify({"status": "already_running", "message": "System is already running"})

    return jsonify({"status": "already_running", "message": "System is already running"})

@app.route('/api/pause', methods=['POST'])
def pause_system():
    """停止视频处理线程。"""
    global system_running, processing_thread
    if system_running:
        system_running = False # Signal the thread to stop
        data_logger.stop_gaze_log() # 停止记录日志
        # Wait for thread to finish? Optional, depends on desired behavior
        # if processing_thread is not None and processing_thread.is_alive():
        #     processing_thread.join(timeout=2.0) # Wait max 2 seconds
        #     if processing_thread.is_alive():
        #          logger.warning("Processing thread did not stop gracefully.")
        # processing_thread = None # Reset thread variable
        logger.info("系统暂停指令已发送")
        return jsonify({"status": "success", "message": "System paused"})

    return jsonify({"status": "already_paused", "message": "System is already paused"})


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取当前系统状态和处理数据。"""
    with data_lock:
        # Create a copy to avoid returning the shared object directly
        status_data = last_processed_data.copy()
    status_data['system_running'] = system_running
    # Add more status info if needed
    return jsonify(status_data)

@app.route('/api/start_calibration', methods=['POST'])
def start_calibration():
    """开始校准流程。"""
    if not system_running:
         return jsonify({"status": "error", "message": "System not running"}), 400
    if calibrator.is_calibrating:
        return jsonify({"status": "warning", "message": "Already calibrating"})

    first_point = calibrator.start_calibration()
    if first_point:
        # Also set the points for error analyzer validation
        error_analyzer.set_calibration_points(calibrator.points_to_calibrate)
        error_analyzer.reset() # Clear previous errors
        logger.info("校准流程启动")
        return jsonify({"status": "success", "message": "Calibration started", "next_point": first_point, "state": calibrator.get_state()})
    else:
        logger.error("无法启动校准")
        return jsonify({"status": "error", "message": "Failed to start calibration"}), 500

@app.route('/api/add_calibration_point', methods=['POST'])
def add_calibration_point_api():
    """前端通知后端记录当前点的校准数据。"""
    if not system_running or not calibrator.is_calibrating:
        return jsonify({"status": "error", "message": "Not in calibration mode or system stopped"}), 400

    # 获取最新的特征数据
    features = None
    with data_lock:
        # --- 开始添加调试日志 ---
        logger.debug(f"API add_calibration_point: 检查 last_processed_data: {bool(last_processed_data)}")
        if last_processed_data:
            logger.debug(f"API add_calibration_point: face_detected = {last_processed_data.get('face_detected')}")
            logger.debug(
                f"API add_calibration_point: 'iris_centers' in last_processed_data = {'iris_centers' in last_processed_data}")
            if 'iris_centers' in last_processed_data:
                logger.debug(f"API add_calibration_point: iris_centers data = {last_processed_data['iris_centers']}")
        # --- 结束添加调试日志 ---

        if last_processed_data and last_processed_data.get('face_detected'):
            if 'iris_centers' in last_processed_data:
                temp_tracking_data = {'iris_centers': last_processed_data['iris_centers']}
                try:  # 添加 try-except 捕获 _extract_features 可能的内部错误
                    features = gaze_estimator._extract_features(temp_tracking_data)
                    logger.debug(f"API add_calibration_point: _extract_features 返回: {features}")  # 记录返回值
                except Exception as e:
                    logger.error(f"API add_calibration_point: 调用 _extract_features 时出错: {e}", exc_info=True)
                    features = None  # 确保出错时 features 为 None
            else:
                logger.debug("API add_calibration_point: last_processed_data 中缺少 iris_centers")
        else:
            logger.debug("API add_calibration_point: 未检测到人脸或 last_processed_data 为空")

    # with data_lock:
    #     if last_processed_data and last_processed_data.get('face_detected'):
    #          # Need to re-extract features based on latest tracking data
    #          # This assumes last_processed_data holds enough info, which might not be ideal
    #          # A better way might be to trigger feature extraction directly here
    #          # For now, let's try using the last iris data if available
    #          if 'iris_centers' in last_processed_data:
    #               # We need the GazeEstimator to extract features consistently
    #               # Let's re-run feature extraction using the data we have
    #               temp_tracking_data = {'iris_centers': last_processed_data['iris_centers']}
    #               # head_pose = last_processed_data.get('head_pose') # If head pose was stored
    #               features = gaze_estimator._extract_features(temp_tracking_data) # Pass necessary args

    if features is not None:
        if calibrator.add_calibration_point(features):
            logger.info(f"API: 成功记录校准点 {calibrator.current_point_index + 1} 的数据")
            return jsonify({"status": "success", "message": "Calibration data added", "state": calibrator.get_state()})
        else:
            logger.warning(f"API: 添加校准数据失败 (可能间隔过短或特征无效)")
            return jsonify({"status": "warning", "message": "Failed to add calibration data (maybe too fast or no features)"})
    else:
        logger.warning("API: 无法获取有效特征来记录校准点")
        return jsonify({"status": "error", "message": "Could not get valid features for calibration point"}), 400


@app.route('/api/next_calibration_point', methods=['POST'])
def next_calibration_point_api():
    """移动到下一个校准点。"""
    if not system_running or not calibrator.is_calibrating:
         return jsonify({"status": "error", "message": "Not in calibration mode or system stopped"}), 400

    next_point = calibrator.next_point()
    if next_point:
        logger.info(f"API: 移动到下一个校准点 {calibrator.current_point_index + 1}")
        return jsonify({"status": "success", "message": "Moved to next calibration point", "next_point": next_point, "state": calibrator.get_state()})
    else:
        # Reached the end or error
        logger.info("API: 所有校准点数据采集完成，准备训练。")
        # Automatically finish and train
        model_params = calibrator.finish_calibration()
        if model_params:
            gaze_estimator.load_calibration_model(model_params)
            # Save the trained model
            data_logger.save_calibration_data(model_params)
            logger.info("API: 校准完成并成功训练/加载模型")
            return jsonify({"status": "finished", "message": "Calibration finished and model trained/loaded.", "state": calibrator.get_state(), "is_calibrated": gaze_estimator.is_calibrated})
        else:
            logger.error("API: 校准数据采集完成，但模型训练失败。")
            return jsonify({"status": "error", "message": "Calibration data collected, but model training failed.", "state": calibrator.get_state(), "is_calibrated": gaze_estimator.is_calibrated}), 500


@app.route('/api/reset', methods=['POST'])
def reset_system():
    """重置系统状态，例如清除校准。"""
    global system_running, last_processed_data, last_frame
    was_running = system_running
    if system_running:
         pause_system() # Stop processing first

    # Reset components
    calibrator.is_calibrating = False
    calibrator.current_point_index = -1
    calibrator.calibration_data = []
    gaze_estimator.is_calibrated = False
    gaze_estimator.calibration_model = None
    error_analyzer.reset()
    face_tracker.blink_count = 0 # Reset blink count
    # Optionally clear logs or specific data
    with data_lock:
        last_processed_data = {}
    with frame_lock:
        last_frame = None # Clear last frame

    logger.info("系统状态已重置")
    # Optionally restart if it was running before
    # if was_running:
    #     start_system()
    return jsonify({"status": "success", "message": "System reset"})


@app.route('/api/save_data', methods=['POST'])
def save_data():
    """触发保存当前数据（例如校准模型）。"""
    # Currently, calibration model is saved automatically after training.
    # This endpoint could be used for other data or manual saves.
    # For now, just confirm save of calibration data if available.
    if gaze_estimator.is_calibrated and gaze_estimator.calibration_model:
         if data_logger.save_calibration_data(gaze_estimator.calibration_model):
              return jsonify({"status": "success", "message": "Calibration model saved."})
         else:
              return jsonify({"status": "error", "message": "Failed to save calibration model."}), 500
    else:
         return jsonify({"status": "warning", "message": "No calibration model to save."})


@app.route('/api/get_error_viz', methods=['GET'])
def get_error_viz_data():
     """获取用于误差可视化的数据。"""
     viz_type = request.args.get('type', 'heatmap') # 'heatmap' or 'vector'
     data = error_analyzer.get_error_visualization_data(type=viz_type)
     stats = error_analyzer.calculate_statistics()
     return jsonify({"visualization_data": data, "statistics": stats})


# --- 异常处理 ---
@app.errorhandler(Exception)
def handle_exception(e):
    """全局异常处理器，返回 JSON 错误信息。"""
    logger.error(f"未捕获的异常: {e}", exc_info=True)
    # 在生产环境中，避免暴露过多细节
    # import traceback
    # tb_str = traceback.format_exc()
    response = {
        "status": "error",
        "message": "An internal server error occurred.",
        # "details": str(e), # Optional: for debugging
        # "traceback": tb_str # Optional: for debugging
    }
    # 根据异常类型可以返回不同的状态码
    # if isinstance(e, ValueError):
    #     return jsonify(response), 400
    return jsonify(response), 500


# --- 主程序入口 ---
if __name__ == '__main__':
    # 使用 waitress 或 gunicorn 等生产级 WSGI 服务器部署
    # app.run(debug=True, host='0.0.0.0', port=5000) # Flask 开发服务器，debug=True 会运行两个实例，小心初始化问题
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True) # threaded=True 允许并发请求处理视频流和API
