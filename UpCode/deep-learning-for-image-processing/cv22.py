# import cv2

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOCUS, 1000)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cv2.imshow('frame', frame)
    
#     if cv2.waitKey(1) == ord('q'):
#         break




import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', '2'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# fps = cap.get(cv2.CAP_PROP_FOURCC)
# print(fps)

# cap.set(cv2.CAP_PROP_FOCUS, 1)
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)

while True:
	ret, frame = cap.read()
	if not ret:
		break
    
	cv2.imshow('frame', frame)
	if cv2.waitKey(30) == ord('q'):
		break







# import cv2

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("摄像头未成功打开")
#     exit()


# supported_resolutions = []
# for width in range(320, 1921, 320): 
#     for height in range(240, 1081, 240): 
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#         actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         if actual_width == width and actual_height == height:
#             supported_resolutions.append((width, height))

# print("支持的分辨率设置:", supported_resolutions)
