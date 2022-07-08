import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from natsort import natsorted
import platform
import time

img = None
drawing = False

height = 0
width = 0
count = 0
added_pixels = 50

file_path = os.path.dirname(os.path.abspath(__file__))

x1 = np.zeros((2,), dtype=np.uint8)
x2 = np.zeros((2,), dtype=np.uint8)

save_data = '/Users/dominik/Desktop/TODO/GroundTruth.txt'

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

		print("x1: {}, x2: {}, height: {}, width: {}".format(x1 - added_pixels, x2 - added_pixels, x2[0] - x1[0], x2[1] - x1[1]))
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

		print("save coordinates? (y/N)")
		save_coordinates = input()

		if save_coordinates == 'y' or save_coordinates == 'Y':
			with open(save_data, 'a') as file:
				values = "1 " + str(int(x1[0]) - added_pixels) + " " + str(int(x1[1]) - added_pixels) + " " + str(int(height)) + " " + str(int(width)) + "\n"
				file.write(values)

			cv2.destroyWindow("image")
			print()


def get_files(dir_path):
	res = []
	for path in os.listdir(dir_path):
		if os.path.isfile(os.path.join(dir_path, path)):
			if platform.system() == "Windows":
				res.append(dir_path + "\\" + path)
			else:
				res.append(dir_path + "/" + path)

	return natsorted(res)[1:]


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
	global img, lst_img, count
	count = 0
	start = 51	# minus eins, weil wir in der datei bei 1 anfangen und nicht bei null
							# minus eins, weil eine datei bei 136 vergessen wurde

	cv2.namedWindow("image")
	if os.environ == "Windows":
		path = r"C:\Users\marin\Desktop\Robotikseminar\Learnings\TrafficSigns\Data\SearchTrafficSigns\ImageSearch"
	else:
		path = "/Users/dominik/Desktop/TODO"
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