import cv2
import numpy as np
import pickle
from time import time 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()

recogniser.read('train_data.yml')
with open('label_ids.pickle', 'rb') as f:
	id_labels = {v:k for k, v in pickle.load(f).items()}

cap = cv2.VideoCapture(0) #cv2.CAP_DSHOW
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#save = cv2.VideoWriter('out.mp4', fourcc, 30, (640,480), True)

def detect_face_eyes(frame, fps):
	grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	sum = np.sum(grey_frame)
	total_pix = np.shape(grey_frame)
	avg = sum // (total_pix[0] * total_pix[1])
	shp_amount = 0
	if avg < 125:
		shp_amount = 150/(avg + 1)
		shp_amount = shp_amount.round(4)
		if shp_amount > 5:
			shp_amount = 5
		#print(shp_amount)
		grey_frame = brighten_image(grey_frame, amount=shp_amount)
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.65
	color = (255, 255, 255)
	thickness = 2
	faces = face_cascade.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
	if type(faces) != tuple:
		#faces = profile_face_cascade.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
		#if type(faces) == tuple:
		#	grey_frame = cv2.flip(grey_frame, 1)
		#	faces = profile_face_cascade.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
		#	grey_frame = cv2.flip(grey_frame, 1)
		for (x, y, w, h) in faces:
			#print(faces)
			roi_face_grey = grey_frame[y:y+h, x:x+w]
			roi_face_colour = frame[y:y+h, x:x+w]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.rectangle(frame, (x - 1, y+h), (x + w + 1, y + h + 20), (0, 255, 0), cv2.FILLED)
			f_id, conf = recogniser.predict(roi_face_grey)
			conf =  100*(1-conf/300)
			if conf >= 65:
				cv2.putText(frame, id_labels[f_id] + ':' + str(round(conf, 2)) + '%', (x + 2, y + h + 15) , font,  fontScale, color, thickness, cv2.LINE_AA)
				#print(id_labels[f_id], conf)
			#eyes = eye_cascade.detectMultiScale(roi_face_grey)
			#for (x, y, w, h) in eyes:
				#print(eyes)
			#	cv2.rectangle(roi_face_colour, (x, y), (x + w, y + h), (255, 255, 255), 1)
	cv2.putText(frame, "FPS: " + str(fps), (10,20), font,  fontScale, color, thickness, cv2.LINE_AA)
	cv2.putText(frame, str(shp_amount), (frame.shape[1] - 70,20), font,  fontScale, color, thickness, cv2.LINE_AA)
	return frame #cv2.flip(frame, 1)

def brighten_image(image, gamma=0.4, kernal_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
	lookUpTable = np.empty((1,256), np.uint8)
	for i in range(256):
		lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
	image = cv2.LUT(image, lookUpTable)
	blurred = cv2.GaussianBlur(image, kernal_size, sigma)
	sharpened = float(amount + 1) * image - float(amount) * blurred
	sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
	sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
	sharpened = sharpened.round().astype(np.uint8)
	if threshold > 0:	
		low_contrast_mask = np.absolute(image - blurred) < threshold
		np.copyto(sharpened, image, where=low_contrast_mask)
	return sharpened

start_time = time()
end_time = start_time
n_frames = 0
fps  = 0
while cap.isOpened():
	if n_frames >= 8: 
		fps = n_frames//(end_time - start_time)
		start_time = end_time
		n_frames = 0
	try:
		ret, frame = cap.read()
		if not ret :
			print("cam problem")
			break
		frame = cv2.flip(frame, 1)
		cv2.imshow('frame', detect_face_eyes(frame, fps))
		n_frames += 1
		end_time = time()
		#save.write(frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	except Exception as e:
		print('force closing :', e)
		cap.release()
		#save.release()
		cv2.destroyAllWindows()
		quit()

print('closing')


#save.release()
cap.release()
cv2.destroyAllWindows()
for i in range (1,5):
	cv2.waitKey(1)