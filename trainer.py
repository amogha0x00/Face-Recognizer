import cv2
import numpy as np
import pickle
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, 'images')

def get_pics(label, freq):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
	print('Taking Pics of', label)
	c_id = 0
	label = os.path.join(image_dir, label)
	if not os.path.exists(label):
		os.mkdir(label)
	else:
		file_list = os.listdir(label)
		if file_list:
			c_id = (max([int(f[:f.index('.')]) for f in file_list]) + 1) * freq
	print(c_id)
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		try:
			a, frame = cap.read()
			if not a :
				print("cam problem")
				continue
			frame = cv2.flip(frame, 1)
			grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			face = face_cascade.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
			if type(face) == tuple:
				face = profile_face_cascade.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
				#if type(face) == tuple:
				#	grey_frame = cv2.flip(grey_frame, 1)
				#	face = profile_face_cascade.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
				#	grey_frame = cv2.flip(grey_frame, 1)

			for (x, y, w, h) in face:
				#print(face)
				roi_face_grey = grey_frame[y:y+h, x:x+w]
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				if roi_face_grey.any():
					if not (c_id % freq):
						print('writing : ', c_id//freq)
						cv2.imwrite(os.path.join(label, str(c_id//freq) + '.jpg'), roi_face_grey)
					c_id += 1

			cv2.imshow('frame', frame)
			
			#save.write(frame)
			key = cv2.waitKey(10) & 0xFF
			if key == ord('q'):
				raise Exception
			elif key == ord('t'):
				cap.release()
				cv2.destroyAllWindows()
				train_with_images()
				print('Done Training!!!!')
				quit()
		except Exception as e:
			print('force closing :', e)
			cap.release()
			cv2.destroyAllWindows()
			quit()

def train_with_images():
	recogniser = cv2.face.LBPHFaceRecognizer_create()
	train_roi = []
	train_id = []
	label_ids = {}
	label_id = 0
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith('png') or file.endswith('jpg'):
				img_path = os.path.join(root, file) 
				print(img_path)
				label = os.path.basename(root).replace(' ', '_')
				if not label in label_ids:
					label_ids[label] = label_id
					label_id += 1
				current_id = label_ids[label]
				roi = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
				#face = face_cascade.detectMultiScale(img)
				#for (x, y, w, h) in face:
				#roi = img[y:y+h, x:x+w]
				shape = roi.shape
				if (shape[0] + shape[1])//2 > 125:
					roi = cv2.resize(roi, (125, 125))
				train_roi.append(roi)
				train_id.append(current_id)
				#print(train_id)
	
	with open('label_ids.pickle', 'wb') as f:
		pickle.dump(label_ids, f)
	recogniser.train(train_roi, np.array(train_id))
	recogniser.save('train_data.yml')

if __name__ == '__main__':
	choice = input("Get new images(G) or Train with existing images(T) (G/T): " ) 
	if choice.lower() == 'g':
		label = input("Input the label: ")
		print("press (q) to exit or (t) to train")
		try:
			get_pics(label, 3)
		except KeyboardInterrupt:
			print("Quiting")
	else:
		print("Begining Training...")
		train_with_images()
