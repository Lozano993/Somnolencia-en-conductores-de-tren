## Create dataset

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
import os
import shutil

video_path = "D:/Guisella/Python/yolo/Dataset/Frames/VID-20240915-WA0053.mp4"
folder = video_path.split("/")[-1].split(".")[0]
print(folder)
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
os.mkdir(f"{folder}/drowsy")
os.mkdir(f"{folder}/awake")
cap = cv2.VideoCapture(video_path)

detector = FaceMeshDetector(maxFaces=1)
detectorHand = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
idList = [0, 15, 76, 306, 159, 145, 130, 243, 386, 374, 362, 263]
ratioList = []
Drowsycounter = 0
Awakecounter = 0
color = (255,0,255)
frames = 0
if not cap.isOpened():
    print("Error al abrir el video")
    exit()
    

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    frame = img.copy()
    frames = frames + 1
    if frames % 6 != 0:
        continue
    img,faces = detector.findFaceMesh(img, draw=False)
    hands, img = detectorHand.findHands(img, draw=False, flipType=True)

    if hands:
        Drowsycounter += 1
        cv2.imwrite(f"{folder}/drowsy/{folder}_{Drowsycounter:04d}.jpg", frame)
        #print("Drowsy")
    elif faces:
        face = faces[0]
        
        '''
        for idx, point in enumerate(face):
            frame = img.copy()
            cv2.circle(frame, point, 1, color, cv2.FILLED)
            cv2.putText(frame, str(idx), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow("Image", frame)
            if cv2.waitKey(0) & 0xFF == ord('w'):
                break
        for id in idList:
            cv2.circle(img,face[id],1,color,cv2.FILLED)
        '''

        up = face[idList[4]]
        down = face[idList[5]]
        left = face[idList[6]]
        right = face[idList[7]]
        verLength, _ = detector.findDistance(up,down) #vertical length
        horLength, _ = detector.findDistance(left,right)#horizontal length
        ratioLeft = int((verLength/horLength)*100)
        
        up = face[idList[8]]
        down = face[idList[9]]
        left = face[idList[10]]
        right = face[idList[11]]
        verLength, _ = detector.findDistance(up,down) #vertical length
        horLength, _ = detector.findDistance(left,right)#horizontal length
        ratioRight = int((verLength/horLength)*100)
        
        up = face[idList[0]]
        down = face[idList[1]]
        left = face[idList[2]]
        right = face[idList[3]]
        verLength, _ = detector.findDistance(up,down) #vertical length
        horLength, _ = detector.findDistance(left,right)#horizontal length
        ratioMouth = int((verLength/horLength)*100)

        #print(f"{ratioLeft}, {ratioRight}, {ratioMouth}")

        if ratioLeft < 18 or ratioRight < 18 or ratioMouth > 35: 
            Drowsycounter += 1
            color = (0,200,0)
            cvzone.putTextRect(img,f'Blink Count:{Drowsycounter}',(100,100), colorR= color)
            cv2.imwrite(f"{folder}/drowsy/{folder}_{Drowsycounter:04d}.jpg", frame)
            #print("Drowsy")
        else:
            Awakecounter += 1
            cv2.imwrite(f"{folder}/awake/{folder}_{Awakecounter:04d}.jpg", frame)
            #print("Awake")
            
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print(f"Finish, {Drowsycounter:04d}, {Awakecounter:04d}")