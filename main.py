import cv2
import os
import mediapipe as mp
from playsound import playsound
import argparse

parser = argparse.ArgumentParser(description='Triggers an alarm to help break the habit of scratching your nose')
parser.add_argument('--preview', action='store_true', default=False, help='Enable preview')
parser.add_argument('--mute', action='store_true', default=False, help='Mute sound')
args = parser.parse_args()

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh()
alert_path = os.path.join(os.path.dirname(__file__), "media/stop-it.m4a")

def is_touching_nose(hand_results, face_results, frame_shape):
    if not hand_results.multi_hand_landmarks or not face_results.multi_face_landmarks:
        return False

    nose = face_results.multi_face_landmarks[0].landmark[1]
    nose_x, nose_y = int(nose.x * frame_shape[1]), int(nose.y * frame_shape[0])

    for hand_landmarks in hand_results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            hand_x, hand_y = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
            if abs(nose_x - hand_x) < 50 and abs(nose_y - hand_y) < 50:
                return True

    return False

def play_alert_sound():
    playsound(alert_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    hand_results = hands.process(image)
    face_results = face_mesh.process(image)

    if is_touching_nose(hand_results, face_results, frame.shape):
        if not args.mute:
            play_alert_sound()
        if args.preview:
            cv2.putText(frame, "Don't touch your nose!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if args.preview:
        cv2.imshow('Face and Hand Tracking', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()