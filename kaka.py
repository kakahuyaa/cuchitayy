import cv2
import mediapipe as mp
import numpy as np
import os

# Khởi tạo các đối tượng Mediapipe và OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Hàm để tải dữ liệu cử chỉ từ thư mục
def load_gesture_data(data_folder):
    gesture_data = {}
    for gesture_folder in os.listdir(data_folder):
        gesture_label = gesture_folder  # Lấy tên thư mục làm nhãn
        gesture_data[gesture_label] = []
        folder_path = os.path.join(data_folder, gesture_folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        gesture_data[gesture_label].append(hand_landmarks)
    return gesture_data

# Hàm để so sánh cử chỉ
def compare_gesture(hand_landmarks, gesture_data):
    current_gesture = np.array([hand_landmarks.landmark[i].y for i in range(21)])  # Lấy 21 điểm
    recognized_gesture = 'Unknown'
    min_distance = float('inf')
    for gesture_label, landmarks_list in gesture_data.items():
        for gesture_landmarks in landmarks_list:
            gesture_points = np.array([gesture_landmarks.landmark[i].y for i in range(21)])
            distance = np.linalg.norm(current_gesture - gesture_points)
            if distance < min_distance:
                min_distance = distance
                recognized_gesture = gesture_label
    return recognized_gesture

# Tạo từ điển ánh xạ từ số đến chữ cái
number_to_letter = {
    '0': 'A',
    '1': 'B',
    '2': 'C',
    '3': 'D',
    '4': 'E',
    '5': 'F',
    '6': 'G',
    '7': 'H',
    '8': 'I',
    '9': 'J',
    '10': 'K',
    '11': 'L',
    '12': 'M',
    '13': 'N',
    '14': 'O',
    '15': 'P',
    '16': 'Q',
    '17': 'R',
    '18': 'S',
    '19': 'T',
    '20': 'U',
    '21': 'V',
    '22': 'W',
    '23': 'X',
    '24': 'Y',
    '25': 'Z'
}

# Khởi động camera
cap = cv2.VideoCapture(0)

# Tải dữ liệu cử chỉ từ thư mục
data_folder = 'data'  # Thay đổi đường dẫn tới thư mục chứa dữ liệu cử chỉ
gesture_data = load_gesture_data(data_folder)

while True:
    success, img = cap.read()
    if not success:
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Nhận diện cử chỉ và hiển thị chữ cái
            gesture = compare_gesture(hand_landmarks, gesture_data)
            letter = number_to_letter.get(gesture, 'Unknown')  # Lấy chữ cái tương ứng
            cv2.putText(img, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    # Hiển thị hình ảnh
    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()