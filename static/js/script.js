document.addEventListener('DOMContentLoaded', () => {
    const videoFeed = document.getElementById('videoFeed');
    const gazePointMarker = document.getElementById('gazePointMarker');
    const calibrationTarget = document.getElementById('calibrationTarget');
    const videoContainer = document.getElementById('videoContainer');

    const startButton = document.getElementById('startButton');
    const pauseButton = document.getElementById('pauseButton');
    const resetButton = document.getElementById('resetButton');
    const startCalibrationButton = document.getElementById('startCalibrationButton');
    const saveDataButton = document.getElementById('saveDataButton');
    // const getErrorVizButton = document.getElementById('getErrorVizButton');

    const systemStatus = document.getElementById('systemStatus');
    const fpsDisplay = document.getElementById('fpsDisplay');
    const resolutionDisplay = document.getElementById('resolutionDisplay');
    const faceDetectedStatus = document.getElementById('faceDetectedStatus');
    const calibrationStatus = document.getElementById('calibrationStatus');
    const blinkCountDisplay = document.getElementById('blinkCountDisplay');
    const gazeDirectionDisplay = document.getElementById('gazeDirectionDisplay');
    const networkLatencyDisplay = document.getElementById('networkLatencyDisplay');
    const lastMessage = document.getElementById('lastMessage');

    const calibrationGuide = document.getElementById('calibrationGuide');
    const calibrationInstructions = document.getElementById('calibrationInstructions');
    const calibrationProgress = document.getElementById('calibrationProgress');
    const collectDataButton = document.getElementById('collectDataButton');
    const nextPointButton = document.getElementById('nextPointButton');
    const finishCalibrationButton = document.getElementById('finishCalibrationButton'); // Maybe auto-finish

    const heatmapContainer = document.getElementById('heatmapContainer');
    const errorStats = document.getElementById('errorStats');
    let heatmapInstance = null;


    let statusInterval = null;
    let isSystemRunning = false;
    let isCalibrating = false;
    let currentCalibrationState = null;
    let lastStatusData = {}; // Store last received status

    const API_BASE_URL = '/api'; // Or full URL if needed

    // --- API Helper Function ---
    async function callApi(endpoint, method = 'GET', body = null) {
        const startTime = performance.now();
        const url = `${API_BASE_URL}${endpoint}`;
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
        };
        if (body && method !== 'GET') {
            options.body = JSON.stringify(body);
        }

        try {
            const response = await fetch(url, options);
            const endTime = performance.now();
            const latency = Math.round(endTime - startTime);
            networkLatencyDisplay.textContent = `网络延迟 (API): ${latency} ms`;

            const data = await response.json();
            if (!response.ok) {
                console.error(`API Error (${response.status}):`, data.message || 'Unknown error');
                lastMessage.textContent = `错误: ${data.message || '请求失败'}`;
                lastMessage.style.color = 'red';
                // Handle specific errors if needed
                if (response.status === 500) {
                     alert(`服务器内部错误: ${data.message}`);
                }
                return null; // Indicate failure
            }
            // Display success messages?
            if (data.message) {
                lastMessage.textContent = `消息: ${data.message}`;
                lastMessage.style.color = 'green';
            }
            return data; // Return successful data
        } catch (error) {
            const endTime = performance.now();
            const latency = Math.round(endTime - startTime);
            networkLatencyDisplay.textContent = `网络延迟 (API): ${latency} ms`;
            console.error('Network or API call failed:', error);
            lastMessage.textContent = '错误: 网络或API调用失败';
            lastMessage.style.color = 'red';
            return null; // Indicate failure
        }
    }

    // --- UI Update Functions ---
    function updateUIState() {
        startButton.disabled = isSystemRunning;
        pauseButton.disabled = !isSystemRunning;
        startCalibrationButton.disabled = !isSystemRunning || isCalibrating; // Disable if running or calibrating
        resetButton.disabled = isSystemRunning; // Maybe allow reset while running? Let's disable for now.
        saveDataButton.disabled = !isSystemRunning; // Or maybe allow saving anytime?

        if (isSystemRunning) {
            systemStatus.textContent = '状态: 运行中';
            systemStatus.style.color = 'green';
            startStatusUpdates();
        } else {
            systemStatus.textContent = '状态: 已停止';
            systemStatus.style.color = 'red';
            stopStatusUpdates();
            // Reset displays when stopped
            fpsDisplay.textContent = 'FPS: --';
            resolutionDisplay.textContent = '分辨率: --';
            faceDetectedStatus.textContent = '人脸检测: No';
            blinkCountDisplay.textContent = '眨眼次数: 0';
            gazeDirectionDisplay.textContent = '注视方向: --';
            gazePointMarker.style.display = 'none';
            networkLatencyDisplay.textContent = '网络延迟 (API): -- ms';
            lastMessage.textContent = '消息: --';
            calibrationStatus.textContent = `校准状态: ${lastStatusData.is_calibrated ? '已校准' : '未校准'}`; // Show last known state
             // Hide calibration UI if system stops
             if(isCalibrating) endCalibrationUI(false); // Force end UI if system stops
        }
    }

    function updateStatusDisplays(data) {
        lastStatusData = data; // Store the latest data
        fpsDisplay.textContent = `FPS: ${data.fps !== undefined ? data.fps.toFixed(1) : '--'}`;
        resolutionDisplay.textContent = `分辨率: ${data.resolution || '--'}`;
        faceDetectedStatus.textContent = `人脸检测: ${data.face_detected ? 'Yes' : 'No'}`;
        blinkCountDisplay.textContent = `眨眼次数: ${data.blink_count !== undefined ? data.blink_count : '--'}`;
        gazeDirectionDisplay.textContent = `注视方向: ${data.gaze_direction || '--'}`;
        calibrationStatus.textContent = `校准状态: ${data.is_calibrated ? '已校准' : '未校准'}`;

        if (data.status_message) {
             lastMessage.textContent = `消息: ${data.status_message}`;
             lastMessage.style.color = data.status_message.includes("错误") ? 'red' : 'orange';
        } else if (lastMessage.textContent.startsWith("消息:")) { // Clear old message if no new one
             // Keep error messages displayed until next API call clears them
             // lastMessage.textContent = '消息: --';
             // lastMessage.style.color = 'black';
        }


        // Update Gaze Marker Position
        if (data.gaze_point && data.gaze_point[0] !== null && data.gaze_point[1] !== null && !isCalibrating) {
            const videoRect = videoContainer.getBoundingClientRect();
            const videoElementRect = videoFeed.getBoundingClientRect(); // Use the img element
            const scaleX = videoElementRect.width / (data.resolution ? parseInt(data.resolution.split('x')[0]) : videoElementRect.width);
            const scaleY = videoElementRect.height / (data.resolution ? parseInt(data.resolution.split('x')[1]) : videoElementRect.height);

            // Calculate position relative to the video container
            const markerX = data.gaze_point[0] * scaleX + (videoElementRect.left - videoRect.left);
            const markerY = data.gaze_point[1] * scaleY + (videoElementRect.top - videoRect.top);

            gazePointMarker.style.left = `${markerX}px`;
            gazePointMarker.style.top = `${markerY}px`;
            gazePointMarker.style.display = 'block';
        } else {
            gazePointMarker.style.display = 'none';
        }

        // Update Calibration State if changed
        if (data.calibration_state) {
             handleCalibrationStateUpdate(data.calibration_state);
             isCalibrating = data.calibration_state.is_calibrating; // Update local flag
        }
    }


    // --- Status Polling ---
    function startStatusUpdates() {
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        // Poll frequently for real-time feel
        statusInterval = setInterval(async () => {
            if (!isSystemRunning) {
                stopStatusUpdates();
                return;
            }
            const data = await callApi('/status');
            if (data) {
                updateStatusDisplays(data);
            } else {
                // Handle error case, maybe stop polling or show error
                console.warn("Failed to get status update.");
            }
        }, 100); // Poll every 100ms
    }

    function stopStatusUpdates() {
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
    }

    // --- Calibration UI Logic ---
    function startCalibrationUI(firstPoint) {
        isCalibrating = true;
        calibrationGuide.style.display = 'block';
        collectDataButton.style.display = 'inline-block';
        nextPointButton.style.display = 'inline-block';
        finishCalibrationButton.style.display = 'none'; // Show only at the end? Or auto-finish.
        gazePointMarker.style.display = 'none'; // Hide gaze marker during calibration

        updateCalibrationUI(currentCalibrationState); // Update with initial state
        moveCalibrationTarget(firstPoint);
    }

     function handleCalibrationStateUpdate(state) {
         currentCalibrationState = state; // Store the latest state
         if (isCalibrating && state.is_calibrating) {
             updateCalibrationUI(state);
             moveCalibrationTarget(state.current_target_point);
         } else if (isCalibrating && !state.is_calibrating) {
             // Calibration finished (likely via API call like next_point reaching end)
             endCalibrationUI(true); // End UI successfully
         }
         // Update overall status display as well
         calibrationStatus.textContent = `校准状态: ${state.is_calibrating ? '进行中' : (lastStatusData.is_calibrated ? '已校准' : '未校准')}`;

     }

    function updateCalibrationUI(state) {
        if (!state || !isCalibrating) return;
        calibrationProgress.textContent = `进度: ${state.current_point_index + 1} / ${state.total_points}`;
        calibrationInstructions.textContent = `请稳定注视屏幕上的蓝色圆点 (${state.current_point_index + 1}/${state.total_points})。准备好后点击“采集数据”按钮。`;
        // Enable/disable buttons based on state?
        collectDataButton.disabled = false; // Allow collecting
        nextPointButton.disabled = false; // Allow moving to next
    }

    function moveCalibrationTarget(point) {
        if (point && point[0] !== null && point[1] !== null) {
            const videoRect = videoContainer.getBoundingClientRect();
            const videoElementRect = videoFeed.getBoundingClientRect();
            const scaleX = videoElementRect.width / (lastStatusData.resolution ? parseInt(lastStatusData.resolution.split('x')[0]) : videoElementRect.width);
            const scaleY = videoElementRect.height / (lastStatusData.resolution ? parseInt(lastStatusData.resolution.split('x')[1]) : videoElementRect.height);

            const targetX = point[0] * scaleX + (videoElementRect.left - videoRect.left);
            const targetY = point[1] * scaleY + (videoElementRect.top - videoRect.top);

            calibrationTarget.style.left = `${targetX}px`;
            calibrationTarget.style.top = `${targetY}px`;
            calibrationTarget.style.display = 'block';
            calibrationTarget.classList.add('animate'); // Add animation
        } else {
            calibrationTarget.style.display = 'none';
            calibrationTarget.classList.remove('animate');
        }
    }

    function endCalibrationUI(success = true) {
        isCalibrating = false;
        calibrationGuide.style.display = 'none';
        collectDataButton.style.display = 'none';
        nextPointButton.style.display = 'none';
        finishCalibrationButton.style.display = 'none';
        calibrationTarget.style.display = 'none';
        calibrationTarget.classList.remove('animate');
        updateUIState(); // Re-enable other buttons

        if(success && lastStatusData.is_calibrated) {
             lastMessage.textContent = "消息: 校准成功完成！";
             lastMessage.style.color = 'blue';
             // Fetch and display error visualization after successful calibration
             fetchErrorVisualization();
        } else if (!success) {
             lastMessage.textContent = "消息: 校准已取消或失败。";
             lastMessage.style.color = 'orange';
        }
    }


    // --- Error Visualization ---
    function initializeHeatmap() {
        if (!heatmapInstance) {
             try {
                heatmapInstance = h337.create({
                    container: heatmapContainer,
                    radius: 30, // Adjust radius as needed
                    maxOpacity: .6,
                    minOpacity: 0,
                    blur: .75
                });
                console.log("Heatmap initialized");
             } catch (e) {
                 console.error("Failed to initialize heatmap:", e);
                 heatmapContainer.innerHTML = "<p style='color:red;'>无法加载热图库。</p>";
             }
        }
    }

    async function fetchErrorVisualization() {
        if (!lastStatusData.is_calibrated) {
             console.log("Cannot fetch error viz: system not calibrated.");
             // Clear previous heatmap?
             if(heatmapInstance) heatmapInstance.setData({ max: 0, data: [] });
             errorStats.textContent = "平均误差: -- px | 标准差: -- px";
             return;
        }

        const vizData = await callApi('/get_error_viz?type=heatmap');
        if (vizData && vizData.visualization_data) {
            initializeHeatmap();
            if (heatmapInstance) {
                const heatmapData = vizData.visualization_data;
                 // Adjust data coordinates relative to the heatmap container if needed
                 // Assuming heatmap covers the same area as the video feed for now
                 const videoRect = videoContainer.getBoundingClientRect();
                 const videoElementRect = videoFeed.getBoundingClientRect();
                 const heatmapRect = heatmapContainer.getBoundingClientRect();

                 const scaleX = heatmapRect.width / (lastStatusData.resolution ? parseInt(lastStatusData.resolution.split('x')[0]) : heatmapRect.width);
                 const scaleY = heatmapRect.height / (lastStatusData.resolution ? parseInt(lastStatusData.resolution.split('x')[1]) : heatmapRect.height);
                 // Offset if heatmap isn't perfectly aligned with video origin
                 const offsetX = videoElementRect.left - heatmapRect.left + (videoElementRect.left - videoRect.left);
                 const offsetY = videoElementRect.top - heatmapRect.top + (videoElementRect.top - videoRect.top);


                 const transformedData = heatmapData.data.map(point => ({
                     x: Math.round(point.x * scaleX + offsetX),
                     y: Math.round(point.y * scaleY + offsetY),
                     value: point.value
                 }));

                heatmapInstance.setData({
                    max: heatmapData.max || 50, // Provide a default max value
                    data: transformedData
                });
                 console.log("Heatmap updated");
            }
            // Update stats text
            if (vizData.statistics) {
                 const stats = vizData.statistics;
                 errorStats.textContent = `平均误差: ${stats.mean_error !== null ? stats.mean_error : '--'} px | 标准差: ${stats.std_dev !== null ? stats.std_dev : '--'} px (${stats.count} samples)`;
            }
        } else {
             console.warn("Failed to fetch error visualization data.");
             if(heatmapInstance) heatmapInstance.setData({ max: 0, data: [] }); // Clear heatmap on error
             errorStats.textContent = "平均误差: -- px | 标准差: -- px";
        }
    }


    // --- Event Listeners ---
    startButton.addEventListener('click', async () => {
        const result = await callApi('/start', 'POST');
        if (result && (result.status === 'success' || result.status === 'already_running')) {
            isSystemRunning = true;
            updateUIState();
            videoFeed.src = `${videoFeed.src.split('?')[0]}?t=${new Date().getTime()}`; // Force reload video stream
        }
    });

    pauseButton.addEventListener('click', async () => {
        const result = await callApi('/pause', 'POST');
        if (result && (result.status === 'success' || result.status === 'already_paused')) {
            isSystemRunning = false;
            updateUIState();
        }
    });

    resetButton.addEventListener('click', async () => {
        if (isSystemRunning) {
            alert("请先暂停系统再进行重置。");
            return;
        }
        if (confirm("确定要重置系统吗？这将清除校准数据和状态。")) {
            const result = await callApi('/reset', 'POST');
            if (result && result.status === 'success') {
                 isSystemRunning = false; // Ensure state is stopped
                 isCalibrating = false;
                 lastStatusData = {}; // Clear local data cache
                 updateUIState();
                 updateStatusDisplays({}); // Clear displays
                 endCalibrationUI(false); // Ensure calibration UI is hidden
                 // Clear heatmap
                 if (heatmapInstance) heatmapInstance.setData({max: 0, data: []});
                 errorStats.textContent = "平均误差: -- px | 标准差: -- px";
                 lastMessage.textContent = "消息: 系统已重置。";
                 lastMessage.style.color = 'blue';
            }
        }
    });

    startCalibrationButton.addEventListener('click', async () => {
        const result = await callApi('/start_calibration', 'POST');
        if (result && result.status === 'success') {
            startCalibrationUI(result.next_point);
            handleCalibrationStateUpdate(result.state); // Process initial state
            updateUIState(); // Update button states
        } else if (result && result.status === 'warning') {
            alert("已经在校准中。");
        } else {
             alert("启动校准失败，请检查后端日志。");
        }
    });

    collectDataButton.addEventListener('click', async () => {
        collectDataButton.disabled = true; // Prevent double clicks
        const result = await callApi('/add_calibration_point', 'POST');
        if (result && result.status === 'success') {
             // Data added, UI might update via status poll, or update manually
             handleCalibrationStateUpdate(result.state);
             calibrationInstructions.textContent = `数据已采集! 点击“下一个点”继续。`;
        } else {
             // Handle error (e.g., too fast, no features)
             alert(`采集数据失败: ${result ? result.message : '未知错误'}`);
        }
        collectDataButton.disabled = false; // Re-enable
    });

    nextPointButton.addEventListener('click', async () => {
        nextPointButton.disabled = true; // Prevent double clicks
        const result = await callApi('/api/next_calibration_point', 'POST');
        if (result) {
             if (result.status === 'success') {
                 // Moved to next point
                 moveCalibrationTarget(result.next_point);
                 handleCalibrationStateUpdate(result.state);
             } else if (result.status === 'finished') {
                 // Calibration finished successfully
                 alert(`校准完成: ${result.message}`);
                 endCalibrationUI(true);
                 handleCalibrationStateUpdate(result.state); // Update final state
                 lastStatusData.is_calibrated = result.is_calibrated; // Update local cache
                 updateStatusDisplays(lastStatusData); // Refresh display
             } else {
                 // Error during finish/training
                 alert(`校准出错: ${result.message}`);
                 endCalibrationUI(false);
                 handleCalibrationStateUpdate(result.state);
             }
        } else {
             alert("请求下一个点失败。");
        }
         nextPointButton.disabled = false; // Re-enable
    });


    saveDataButton.addEventListener('click', async () => {
        const result = await callApi('/save_data', 'POST');
        if (result) {
            alert(result.message); // Show feedback
        }
    });

    // getErrorVizButton.addEventListener('click', fetchErrorVisualization); // Or trigger automatically

    // --- Initial State ---
    updateUIState(); // Set initial button states
    // Try to get initial status when page loads?
     callApi('/status').then(data => {
         if(data) {
            isSystemRunning = data.system_running;
            isCalibrating = data.calibration_state ? data.calibration_state.is_calibrating : false;
            updateStatusDisplays(data);
            updateUIState();
            if (isCalibrating) { // Restore calibration UI if page reloads during calibration
                 startCalibrationUI(data.calibration_state.current_target_point);
                 handleCalibrationStateUpdate(data.calibration_state);
            }
            if (data.is_calibrated) {
                 fetchErrorVisualization(); // Fetch heatmap if already calibrated
            }
         }
     });

});