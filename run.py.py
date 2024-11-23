## Run application

import cv2
from ultralytics import YOLO

video_path = "D:/Guisella/Python/yolo/Dataset/Final test/VID-20240919-WA0017.mp4"
model = YOLO("D:/Guisella/Python/yolo/v8n224_2/weights/best.pt")


threshold = 25  # frames
patience = 45  # frames


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video")
    exit()
drowsyCounter = 0
valPatience = patience
patOn = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False, device="cpu")
    speed = int(
        results[0].speed["preprocess"]
        + results[0].speed["inference"]
        + results[0].speed["postprocess"]
    )
    if speed > 29:
        nextFrame = 1
    else:
        nextFrame = 33 - speed
    drowsy = results[0].probs.top1  # 0 = awake, 1 = drowsy
    if drowsy:
        drowsyCounter += 1
        if drowsyCounter > threshold:
            patOn = True
            frame = cv2.copyMakeBorder(
                frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=[35, 35, 220]
            )
            frame = cv2.putText(frame, 'Alerta Somnolencia', (50, 50), 2, 1.2, (35, 35, 220), 2, 2)
        else:
            frame = cv2.copyMakeBorder(
                frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=[240, 240, 240]
            )
    elif patience > 0 and patOn:
        patience -= 1
        frame = cv2.copyMakeBorder(
            frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=[35, 35, 220]
        )
        frame = cv2.putText(frame, 'Alerta Somnolencia', (50, 50), 2, 1.2, (35, 35, 220), 2, 2)
    else:
        patience = valPatience
        patOn = False
        drowsyCounter = 0
        frame = cv2.copyMakeBorder(
            frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=[240, 240, 240]
        )
        

    cv2.imshow("YOLO Real-time Drossiness detector", frame)
    if cv2.waitKey(nextFrame) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
