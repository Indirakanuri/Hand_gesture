import cv2
import pyautogui
import mediapipe as mp
import time
import webbrowser
import numpy as np
import winsound

# === Mediapipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_time = 0
cooldown = 1


def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    thumb_tip = hand_landmarks.landmark[tips_ids[0]]
    thumb_ip = hand_landmarks.landmark[pip_ids[0]]
    thumb_up = thumb_tip.x < thumb_ip.x
    count = 0
    for i in range(1, 5):
        tip = hand_landmarks.landmark[tips_ids[i]]
        pip = hand_landmarks.landmark[pip_ids[i]]
        if tip.y < pip.y:
            count += 1
    return thumb_up, count


def play_beep():
    winsound.Beep(1000, 100)


def virtual_keyboard():
    keys = [
        list("QWERTYUIOP"),
        list("ASDFGHJKL"),
        list("ZXCVBNM "),
        ["BACKSPACE", "ENTER"]
    ]
    current_text = ""
    selecting = False
    selected_key = None
    select_start = 0

    cap_kb = cv2.VideoCapture(0)
    hands_kb = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    while True:
        ret, img = cap_kb.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands_kb.process(img_rgb)

        rows = len(keys)
        cols = max(len(row) for row in keys)
        key_w = int(w / (cols + 2))
        key_h = int((h - 150) / (rows + 1))

        start_y = 50
        key_boxes = []
        for i, row in enumerate(keys):
            row_w = len(row) * (key_w + 5)
            start_x = (w - row_w) // 2
            for key in row:
                end_x = start_x + key_w
                end_y = start_y + key_h
                key_boxes.append((key, (start_x, start_y, end_x, end_y)))
                start_x += key_w + 5
            start_y += key_h + 5

        hovered_key = None

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                lm = handLms.landmark
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                cv2.circle(img, (ix, iy), 10, (0, 255, 0), -1)

                for key, (sx, sy, ex, ey) in key_boxes:
                    if sx < ix < ex and sy < iy < ey:
                        hovered_key = key
                        if selected_key != key:
                            selecting = True
                            selected_key = key
                            select_start = time.time()
                        else:
                            if time.time() - select_start > 1:
                                play_beep()
                                if key == "ENTER":
                                    cap_kb.release()
                                    cv2.destroyAllWindows()
                                    query = current_text.replace(" ", "+")
                                    url = f"https://www.youtube.com/results?search_query={query}"
                                    webbrowser.open(url)
                                    return
                                elif key == "BACKSPACE":
                                    current_text = current_text[:-1]
                                else:
                                    current_text += key
                                selected_key = None
                                selecting = False
                        break
                else:
                    selected_key = None
                    selecting = False

        # Draw keys (after detecting hover)
        start_y = 50
        for i, row in enumerate(keys):
            row_w = len(row) * (key_w + 5)
            start_x = (w - row_w) // 2
            for key in row:
                end_x = start_x + key_w
                end_y = start_y + key_h
                if key == hovered_key:
                    color = (0, 255, 0)  # Hover → green
                else:
                    color = (0, 0, 0)
                cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, -1)
                cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

                font_scale = min(key_w, key_h) / 60
                thickness = 2
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = start_x + (key_w - text_size[0]) // 2
                text_y = start_y + (key_h + text_size[1]) // 2
                cv2.putText(img, key, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                start_x += key_w + 5
            start_y += key_h + 5

        cv2.putText(img, current_text, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        cv2.imshow("Virtual Keyboard", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_kb.release()
    cv2.destroyAllWindows()


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    now = time.time()

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            thumb_up, count = count_fingers(handLms)

            if thumb_up and count == 0 and now - last_time > cooldown:
                pyautogui.press("volumeup")
                cv2.putText(img, "Volume Up", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_time = now

            elif not thumb_up and count == 0 and now - last_time > cooldown:
                pyautogui.press("volumedown")
                cv2.putText(img, "Volume Down", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                last_time = now

            elif count == 1 and now - last_time > cooldown:
                pyautogui.press("k")
                cv2.putText(img, "Play/Pause", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                last_time = now

            elif count == 2 and now - last_time > cooldown:
                pyautogui.hotkey("shift", "n")
                cv2.putText(img, "Next Song", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                last_time = now

            elif count == 3 and now - last_time > cooldown:
                pyautogui.hotkey("shift", "p")
                cv2.putText(img, "Previous Song", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                last_time = now

            elif count >= 4 and now - last_time > cooldown:
                cv2.putText(img, "YouTube Search", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Hand Gesture Control", img)
                cv2.waitKey(1)
                time.sleep(1)
                virtual_keyboard()
                cap = cv2.VideoCapture(0)
                last_time = now

    else:
        cv2.putText(img, "Show at least 1 hand", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
