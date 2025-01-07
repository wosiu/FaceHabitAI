import cv2
import os
import mediapipe as mp
from playsound import playsound

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh()
alert_path = os.path.join(os.path.dirname(__file__), "media/japan-eas-alarm-277877.mp3")


def is_touching_nose(hand_results, face_results, frame_shape):
    if not hand_results.multi_hand_landmarks or not face_results.multi_face_landmarks:
        return False

    # 1 is nose index, taking from first found face
    nose = face_results.multi_face_landmarks[0].landmark[1]
    nose_x, nose_y = int(nose.x * frame_shape[1]), int(nose.y * frame_shape[0])

    # Checking all positions of hand landmarks
    for hand_landmarks in hand_results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            hand_x, hand_y = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
            if abs(nose_x - hand_x) < 50 and abs(nose_y - hand_y) < 50:
                 return True

    return False


def play_alert_sound():
    playsound(alert_path)

# Start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Detect hands and faceq
    hand_results = hands.process(image)
    face_results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if is_touching_nose(hand_results, face_results, frame.shape):
        play_alert_sound()
        cv2.putText(image, "Don't touch your nose!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face and Hand Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
