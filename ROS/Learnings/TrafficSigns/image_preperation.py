import os
from tkinter import Toplevel
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from natsort import natsorted
import platform
import time
from datetime import datetime

img = None
drawing = False

height = 0
width = 0
count = 0
added_pixels = 50

x1 = np.zeros((2,), dtype=np.uint8)
x2 = np.zeros((2,), dtype=np.uint8)


file_path = os.path.dirname(os.path.abspath(__file__))
traffic_sign = 5
save_data = file_path + '/Data/Train/' + str(traffic_sign) + "/"

def click_event(event, x, y, flags, params):
	# y ist die row und x ist column
	# Erstelle zwei punkte, erst untere linkere ecke, dann obere rechte ecke
	global count, points, x1, x2, img, save_data, added_pixels

	if count > 1:
		img = lst_img.copy()
		count = 0
		cv2.imshow('image', img)

	if event == cv2.EVENT_LBUTTONDOWN:
		print(y, ' ', x)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(y) + ',' +
					str(x), (x, y), font,
					1, (255, 0, 0), 1)
		cv2.imshow('image', img)

		if count == 0:
			x1 = np.array([y, x], dtype=float)
		elif count == 1:
			x2 = np.array([y, x], dtype=float)

		count +=1

	if count == 2:
		plt.imshow(img)
		fig, ax = plt.subplots(1)
		ax.imshow(img)

		height = x2[0] - x1[0]
		width = x2[1] - x1[1]
		#rectangle needs first column coordinate and then row coordinate
		rect = patches.Rectangle((x1[1], x1[0]), width=width, height=height, linewidth=1,
								 edgecolor='r', facecolor="none")

		ax.add_patch(rect)
		plt.show()
		plt.pause(3)

		plt.figure().clear()
		plt.close()
		plt.cla()
		plt.clf()

		print("save image? (y/N)")
		save_image = input()

		if save_image == 'y' or save_image == 'Y':
			extraction = extract_image(x1, height, width, img)
			
			date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p_%f")
			filename = str(f"filename_{date}" + ".jpg")
			cv2.imwrite(save_data + filename, extraction)

			cv2.destroyWindow("image")
			print()


def extract_image(coordinate, height, width, img):
	top_left_x = int(coordinate[0] + height)		# height is here negative
	top_left_y = int(coordinate[1])

	extraction = img[top_left_x:top_left_x-int(height), 
					 top_left_y:top_left_y+int(width), :]
	return extraction

def get_files(dir_path, visible=True):
	res = []
	for path in os.listdir(dir_path):
		if os.path.isfile(os.path.join(dir_path, path)):
			if platform.system() == "Windows":
				res.append(dir_path + "\\" + path)
			else:
				res.append(dir_path + "/" + path)
	
	if visible:
		res = natsorted(res)[1:]
	else:
		res = natsorted(res)
	return res


def load_image(path):
	return np.asarray(Image.open(path))


def show_image(img):
	cv2.imshow('image', img)
	cv2.setMouseCallback("image", click_event)
	cv2.waitKey(0)


def prepare_image(img, added_pixels=50):
	rows, columns, c = img.shape

	new_img = np.zeros((rows + 2*added_pixels, columns + 2*added_pixels, c), dtype=np.uint8)
	new_img[added_pixels:rows+added_pixels, added_pixels:columns+added_pixels, :] = img.copy()

	return new_img


def main():
	global img, lst_img, count, file_path
	count = 0
	start = 187 - 1 - 1		# minus eins, weil wir in der datei bei 1 anfangen und nicht bei null
							# minus eins, weil eine datei bei 136 vergessen wurde

	cv2.namedWindow("image")
	if os.environ == "Windows":
		path = file_path = r"\Data\SearchTrafficSigns\ImageSearch\Train"
	else:
		path = file_path + r"/Data/SearchTrafficSigns/ImageSearch/Train"
	files = get_files(path)

	for path in files[start:]:
		print("path: ", path)
		img = load_image(path)
		img = prepare_image(img)

		lst_img = img.copy()
		show_image(img)

		cv2.destroyAllWindows()


if __name__ == '__main__':
	main()