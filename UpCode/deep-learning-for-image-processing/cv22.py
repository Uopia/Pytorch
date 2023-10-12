import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOCUS, 1000)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break