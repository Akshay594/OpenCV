import dlib
import cv2
import numpy as np

##link https://drive.google.com/open?id=1OBWIFn0LzBHNHRYAmbW6NKeKOh2h-PwI


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


image = cv2.imread('me.jpg')

rects = detector(image)
coords = np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

# for i, pos in enumerate(coords):
# 	pts = (pos[0, 0], pos[0, 1])
# 	# cv2.putText(image, str(i), pts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
# 	cv2.circle(image, pts, 3, (0, 255, 255), 2)

# cv2.imshow('image', image)
# cv2.waitKey(0)

def lip_dists(landmarks):
	upper_lip = []
	lower_lip = []

	for i in range(50, 53):
		upper_lip.append(landmarks[i])

	for i in range(65, 68):
		lower_lip.append(landmarks[i])

	upper_lip_mean = np.mean(upper_lip, axis=0)
	lower_lip_mean = np.mean(lower_lip, axis=0)

	return int(upper_lip_mean[:, 1]), int(lower_lip_mean[:, 1])


cap = cv2.VideoCapture(0)
while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)
	rects = detector(frame)
	coords = np.matrix([[p.x, p.y] for p in predictor(frame, rects[0]).parts()])
	upper_lip_dist ,lower_lip_dist = lip_dists(coords)

	if abs(upper_lip_dist - lower_lip_dist) > 25:
		cv2.putText(frame, "Stop Yawing you fool.", (55, 55), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
	else:
		cv2.putText(frame, "Not Yawing.", (55, 55), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)


	# for i, pos in enumerate(coords):
	# 	pts = (pos[0, 0], pos[0, 1])

	# 	cv2.circle(frame, pts, 3, (0, 255, 255), 2)

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0XFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

