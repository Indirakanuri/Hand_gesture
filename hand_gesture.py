import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import webbrowser

# Setup
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
last_action_time = 0
cooldown = 1.0  # seconds

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_start = None


def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    thumb_tip = hand_landmarks.landmark[tips[0]]
    thumb_ip = hand_landmarks.landmark[pips[0]]
    thumb_up = thumb_tip.x < thumb_ip.x
    count = 0
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y:
            count += 1
    return thumb_up, count


def virtual_keyboard():
    webbrowser.open("https://www.google.com")
    time.sleep(1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    now = time.time()

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            thumb_up, count = count_fingers(handLms)
            index_finger = handLms.landmark[8]
            index_x = int(index_finger.x * screen_w)
            index_y = int(index_finger.y * screen_h)

            # Move mouse
            if count == 1:
                pyautogui.moveTo(index_x, index_y, duration=0.1)
                cv2.putText(frame, 'Mouse Move', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Perform actions with cooldown
            if now - last_action_time > cooldown:
                if thumb_up and count == 0:
                    pyautogui.press('volumeup')
                    last_action_time = now
                    cv2.putText(frame, 'Volume Up', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif not thumb_up and count == 0:
                    pyautogui.press('volumedown')
                    last_action_time = now
                    cv2.putText(frame, 'Volume Down', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                elif count == 0:
                    pyautogui.click()
                    last_action_time = now
                    cv2.putText(frame, 'Left Click', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                elif count == 2:
                    pyautogui.rightClick()
                    last_action_time = now
                    cv2.putText(frame, 'Right Click', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                elif count == 3:
                    pyautogui.scroll(-200)
                    last_action_time = now
                    cv2.putText(frame, 'Scroll', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif count == 4:
                    pyautogui.hotkey('win', 'd')
                    last_action_time = now
                    cv2.putText(frame, 'Minimize/Restore', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                elif count >= 5:
                    if gesture_start is None:
                        gesture_start = now
                    elif now - gesture_start >= 2.0:
                        pyautogui.press('k')
                        last_action_time = now
                        cv2.putText(frame, 'Play/Pause', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                        gesture_start = None
                else:
                    gesture_start = None

    else:
        gesture_start = None
        cv2.putText(frame, 'Show Hand', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Laptop Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
