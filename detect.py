import torch
import cv2
import numpy as np

model = torch.hub.load("ultralytics/yolov5",'custom',
                     path = 'Carpeta personal/Documentos/best.pt')
cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    detect = model(frame)
    #info = detect.pandas().xyxy[0]  # im1 predictions
    #print(info)

    cv2.imshow("Detector de equipo de seguridad", np.squeeze(detect.render()))

    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()