#!/usr/bin/env python3
"""
基于MediaPipe的手势控制模块 (Multi-Hand Zoom & Rotate)
"""

import cv2
import mediapipe as mp
import math
import time
import threading
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal

class GestureController(QObject):
    """
    手势控制器
    识别手势并发送控制信号
    
    交互逻辑 (根据用户需求修改):
    1. 单手捏合 (Pinch) + 移动 -> 旋转 (Rotate)
    2. 双手捏合 + 距离变化 -> 缩放 (Zoom)
    3. 单手握拳 (Fist) + 移动 -> 平移 (Pan) [保留以便于移动位置]
    """
    
    # 定义控制信号
    move_signal = pyqtSignal(float, float, float) # x, y, z
    zoom_signal = pyqtSignal(float) # amount
    rotate_signal = pyqtSignal(float, float) # heading, pitch
    reset_signal = pyqtSignal()   # 复位视角
    top_view_signal = pyqtSignal() # 俯视视角
    frame_signal = pyqtSignal(object) # 发送图像帧到UI
    
    # 手势枚举
    GESTURE_NONE = "NONE"
    GESTURE_PAN = "PAN"      # 单手握拳 - 平移
    GESTURE_ROTATE = "ROTATE" # 单手捏合 - 旋转
    GESTURE_ZOOM = "ZOOM"    # 双手捏合 - 缩放
    GESTURE_RESET = "RESET"  # V字手势 - 复位
    GESTURE_TOP = "TOP_VIEW" # 竖大拇指 - 俯视
    
    def __init__(self, camera_id=0, show_window=True):
        super().__init__()
        self.camera_id = camera_id
        self.show_window = show_window
        self.is_running = False
        self.thread = None
        self._is_deleted = False
        
        # MediaPipe 初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # 开启双手识别
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 状态变量
        self.last_process_time = 0
        self.process_interval = 0.03 
        
        # 手势状态管理
        self.current_gesture = self.GESTURE_NONE
        self.gesture_history = deque(maxlen=3)
        
        # 坐标/距离记录
        self.prev_coords = {} # {hand_index: (x, y)}
        self.prev_hands_distance = None # 双手模式下的距离
        
        # 平滑因子
        self.smooth_factor = 0.6
        
        # 灵敏度设置
        self.sensitivity_pan = 1500.0
        self.sensitivity_rotate = 3.5
        self.sensitivity_zoom = 8.0 # 距离变化乘数

    def start(self):
        """启动手势识别线程"""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True
        self.thread.start()
        print("[GestureController] 手势控制已启动")

    def stop(self):
        """停止手势识别"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("[GestureController] 手势控制已停止")
    
    def _safe_emit(self, signal, *args):
        """安全发送信号，处理对象已删除的情况"""
        if self._is_deleted or not self.is_running:
            return
        try:
            signal.emit(*args)
        except RuntimeError as e:
            # 对象已被删除或信号已断开
            self._is_deleted = True
            self.is_running = False
            print(f"[GestureController] 信号发送失败，对象已删除: {e}")
        
    def _run_loop(self):
        """主循环"""
        import os
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[GestureController] 无法打开摄像头 {self.camera_id}")
            self.is_running = False
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while self.is_running:
                current_time = time.time()
                if current_time - self.last_process_time < self.process_interval:
                    time.sleep(0.005)
                    continue
                
                self.last_process_time = current_time
                
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                results = self.hands.process(image_rgb)
                image_rgb.flags.writeable = True
                
                # 手势处理逻辑
                status_text = "IDLE"
                if results.multi_hand_landmarks:
                    num_hands = len(results.multi_hand_landmarks)
                    
                    # 双手模式 (Zoom)
                    if num_hands == 2:
                        h1 = results.multi_hand_landmarks[0]
                        h2 = results.multi_hand_landmarks[1]
                        
                        # 检查两只手是否都在捏合
                        if self._is_pinching(h1) and self._is_pinching(h2):
                            self.current_gesture = self.GESTURE_ZOOM
                            status_text = "ZOOM (2 Hands Pinch)"
                            if not self._is_deleted:
                                self._process_zoom(h1, h2)
                        else:
                            # 如果有双手但没同时捏合，重置状态
                            self.current_gesture = self.GESTURE_NONE
                            self.prev_hands_distance = None
                            
                    # 单手模式 (Rotate or Pan)
                    elif num_hands == 1:
                        hand = results.multi_hand_landmarks[0]
                        
                        # 重置双手距离状态
                        self.prev_hands_distance = None
                        
                        if self._is_pinching(hand):
                            self.current_gesture = self.GESTURE_ROTATE
                            status_text = "ROTATE (1 Hand Pinch)"
                            if not self._is_deleted:
                                self._process_rotate(hand)
                        elif self._is_fist(hand):
                            self.current_gesture = self.GESTURE_PAN
                            status_text = "PAN (1 Hand Fist)"
                            if not self._is_deleted:
                                self._process_pan(hand)
                        elif self._is_victory(hand):
                            # V字手势 - 复位
                            if self.current_gesture != self.GESTURE_RESET and not self._is_deleted:
                                self._safe_emit(self.reset_signal)
                            self.current_gesture = self.GESTURE_RESET
                            status_text = "RESET (Victory)"
                            self.prev_coords.clear()
                        elif self._is_thumb_up(hand):
                            # 竖大拇指 - 俯视
                            if self.current_gesture != self.GESTURE_TOP and not self._is_deleted:
                                self._safe_emit(self.top_view_signal)
                            self.current_gesture = self.GESTURE_TOP
                            status_text = "TOP VIEW (Thumb Up)"
                            self.prev_coords.clear()
                        else:
                            self.current_gesture = self.GESTURE_NONE
                            self.prev_coords.clear()
                else:
                    self.current_gesture = self.GESTURE_NONE
                    self.prev_coords.clear()
                    self.prev_hands_distance = None
                    
                # 绘制反馈
                if self.show_window:
                    if results.multi_hand_landmarks:
                        for landmarks in results.multi_hand_landmarks:
                            self._draw_hand(image_rgb, landmarks)
                    
                    # 绘制状态
                    color = (0, 255, 0)
                    if self.current_gesture == self.GESTURE_ZOOM: color = (255, 0, 0)
                    elif self.current_gesture == self.GESTURE_ROTATE: color = (255, 255, 0)
                    elif self.current_gesture == self.GESTURE_PAN: color = (0, 0, 255)
                    elif self.current_gesture == self.GESTURE_RESET: color = (0, 255, 255) # Cyan
                    elif self.current_gesture == self.GESTURE_TOP: color = (255, 0, 255) # Magenta
                    
                    cv2.putText(image_rgb, f"Mode: {status_text}", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # 安全发送信号，检查对象是否仍然有效
                    self._safe_emit(self.frame_signal, image_rgb)
                    if self._is_deleted:
                        break
        except RuntimeError as e:
            # 捕获PyQt5对象已删除的异常
            print(f"[GestureController] 线程异常（对象已删除）: {e}")
            self._is_deleted = True
        except Exception as e:
            # 捕获其他异常
            print(f"[GestureController] 线程异常: {e}")
        finally:
            # 确保资源被释放
            try:
                cap.release()
            except:
                pass
            self.is_running = False
            print("[GestureController] 手势识别线程已退出")

    def _process_zoom(self, h1, h2):
        """处理双手缩放"""
        # 计算两手中心的距离
        c1 = self._get_hand_center(h1)
        c2 = self._get_hand_center(h2)
        
        dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        
        if self.prev_hands_distance is not None:
            # 距离变化
            delta = dist - self.prev_hands_distance
            
            # 过滤微小抖动
            if abs(delta) > 0.002:
                # 距离变大 -> 放大 -> Zoom In (amount > 0)
                # 距离变小 -> 缩小 -> Zoom Out (amount < 0)
                # cesium_widget: positive amount = zoom in (move forward) ? 
                # 通常: move forward = zoom in.
                # 让我们根据之前的逻辑: "正数为放大"
                self._safe_emit(self.zoom_signal, delta * self.sensitivity_zoom * 100)
        
        self.prev_hands_distance = dist
        # 清除单手坐标缓存，避免模式切换时的跳变
        self.prev_coords.clear()

    def _process_rotate(self, hand):
        """处理单手旋转"""
        cx, cy = self._get_hand_center(hand)
        
        # 使用 EMA 平滑
        sx, sy = self._smooth_coord(0, cx, cy)
        
        if 0 in self.prev_coords:
            prev_x, prev_y = self.prev_coords[0]
            dx = sx - prev_x
            dy = sy - prev_y
            
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                # 映射规则:
                # 左右移动 (dx) -> Heading (左右旋转)
                # 上下移动 (dy) -> Pitch (俯仰旋转)
                self._safe_emit(self.rotate_signal, dx * self.sensitivity_rotate, -dy * self.sensitivity_rotate)
        
        self.prev_coords[0] = (sx, sy)

    def _process_pan(self, hand):
        """处理单手平移"""
        cx, cy = self._get_hand_center(hand)
        sx, sy = self._smooth_coord(0, cx, cy)
        
        if 0 in self.prev_coords:
            prev_x, prev_y = self.prev_coords[0]
            dx = sx - prev_x
            dy = sy - prev_y
            
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                # 映射: 手移动方向 = 地图移动方向
                # 手右移 -> dx>0 -> 想要看右边的地图 -> 相机左移
                # 或者：手右移 -> 抓着地图往右拉 -> 地图右移 -> 相机相对左移
                self._safe_emit(self.move_signal, -dx * self.sensitivity_pan, dy * self.sensitivity_pan, 0)
                
        self.prev_coords[0] = (sx, sy)

    def _get_hand_center(self, hand_landmarks):
        """计算手掌中心 (使用 0号点WRIST 和 9号点MIDDLE_MCP 的中点)"""
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        return ((wrist.x + middle_mcp.x) / 2, (wrist.y + middle_mcp.y) / 2)

    def _smooth_coord(self, hand_idx, raw_x, raw_y):
        """坐标平滑"""
        if hand_idx not in self.prev_coords:
            return raw_x, raw_y
            
        prev_x, prev_y = self.prev_coords[hand_idx]
        new_x = prev_x * self.smooth_factor + raw_x * (1 - self.smooth_factor)
        new_y = prev_y * self.smooth_factor + raw_y * (1 - self.smooth_factor)
        return new_x, new_y

    def _is_pinching(self, hand_landmarks):
        """判断是否捏合 (拇指尖和食指尖靠近)"""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + 
                         (thumb_tip.y - index_tip.y)**2 + 
                         (thumb_tip.z - index_tip.z)**2)
        return dist < 0.05

    def _is_fist(self, hand_landmarks):
        """判断是否握拳 (指尖到手腕距离 < 指关节到手腕距离)"""
        wrist = hand_landmarks.landmark[0]
        
        fingers_bent = 0
        # 检查食指、中指、无名指、小指 (Index 8, Middle 12, Ring 16, Pinky 20)
        # 对比指尖(Tip)和近端指关节(PIP - 6, 10, 14, 18)到手腕的距离
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        for t, p in zip(tips, pips):
            dist_tip = self._dist(hand_landmarks.landmark[t], wrist)
            dist_pip = self._dist(hand_landmarks.landmark[p], wrist)
            if dist_tip < dist_pip:
                fingers_bent += 1
                
        # 至少3根手指弯曲算握拳 (允许拇指或食指略微松开)
        return fingers_bent >= 3

    def _is_victory(self, hand_landmarks):
        """判断是否V字手势 (食指中指伸直，其他弯曲)"""
        wrist = hand_landmarks.landmark[0]
        
        # 指尖
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # 指关节
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        ring_pip = hand_landmarks.landmark[14]
        pinky_pip = hand_landmarks.landmark[18]
        
        # 检查伸直/弯曲
        # Index & Middle Extended
        index_ext = self._dist(index_tip, wrist) > self._dist(index_pip, wrist)
        middle_ext = self._dist(middle_tip, wrist) > self._dist(middle_pip, wrist)
        
        # Ring & Pinky Curled
        ring_curl = self._dist(ring_tip, wrist) < self._dist(ring_pip, wrist)
        pinky_curl = self._dist(pinky_tip, wrist) < self._dist(pinky_pip, wrist)
        
        return index_ext and middle_ext and ring_curl and pinky_curl

    def _is_thumb_up(self, hand_landmarks):
        """判断是否竖大拇指"""
        wrist = hand_landmarks.landmark[0]
        
        # 拇指
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3] # 拇指只有一个指间关节
        
        # 其他手指
        tips = [8, 12, 16, 20] # 食指到小指
        pips = [6, 10, 14, 18]
        
        # 拇指指尖要比指关节远(或者高，取决于方向，这里用距离简化)
        # 且拇指指尖要指向上方 (y坐标更小)
        thumb_ext = self._dist(thumb_tip, wrist) > self._dist(thumb_ip, wrist)
        thumb_up = thumb_tip.y < thumb_ip.y # y轴向下增加，所以小的是上面
        
        # 其他手指弯曲
        others_curled = True
        for t, p in zip(tips, pips):
            if self._dist(hand_landmarks.landmark[t], wrist) > self._dist(hand_landmarks.landmark[p], wrist):
                others_curled = False
                break
                
        return thumb_ext and thumb_up and others_curled

    def _dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _draw_hand(self, image, landmarks):
        self.mp_drawing.draw_landmarks(
            image, 
            landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    controller = GestureController()
    controller.move_signal.connect(lambda x, y, z: print(f"Move: {x:.2f}, {y:.2f}, {z:.2f}"))
    controller.zoom_signal.connect(lambda s: print(f"Zoom: {s:.2f}"))
    controller.rotate_signal.connect(lambda h, p: print(f"Rotate: H={h:.2f}, P={p:.2f}"))
    controller.start()
    sys.exit(app.exec_())