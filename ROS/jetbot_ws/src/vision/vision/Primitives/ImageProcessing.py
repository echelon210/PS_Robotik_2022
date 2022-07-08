from collections import deque
from ..Primitives.camera import CameraImage
from ..utils.utils import class_labels
import cv2

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ReplayBuffer(object):
    def __init__(self, buffer_size) -> None:
        self.count = 0
        self.buffer_size = buffer_size
        self.buffer = deque()

    def add(self, value):
        if self.count < self.buffer_size:
            self.buffer.append(value)
            
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(value)

    def get(self):
        self.count -= 1
        return self.buffer.popleft()

    def clear(self):
        self.buffer.clear()
        self.count = 0


class ImageSeperator(CameraImage, ReplayBuffer):
    def __init__(self, n_images = (2, 4), buffer_size=2, visualization=True , optimization=True, 
                 image_width=640, image_height=480, fps=30):
        CameraImage.__init__(self,image_width, image_height, fps)

        if buffer_size != 2:
            buffer_size = 0
            for n in n_images:
                buffer_size += pow(n, 2)  

            if optimization:
                buffer_size /= 2      
            
        ReplayBuffer.__init__(self, int(buffer_size))

        self.n_images = n_images        # number of images that should be spitted
        self.img = None

        self.visualization = visualization
        self.optimization = optimization

    def set_current_frame(self, img):
        self.img = img
    
    def data_collector_seperator(self, values):
        x, y = 0, 0
        for v, number in zip(values, list(self.n_images)):#[:1]):
            h = int(self.image_height / number)
            w = int(self.image_width / number)

            it = 0
            for i in range(number):
                for j in range(number):
                    x = i * h
                    y = j * w
                    
                    if it == v:
                        ext_img = self.extract_image(x, y, w, h)
                        if isinstance(ext_img, np.ndarray):
                            self.add(ext_img)
                    
                    it += 1
    
    def add_extracted_img(self, i, j, w, h):
        x = i * h
        y = j * w
        ext_img = self.extract_image(x, y, w, h)

        if isinstance(ext_img, np.ndarray):
            self.add(ext_img)
            # self.save_img(ext_img)

        # self.save_img(self.img)

    def save_img(self, img):
        date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p_%f")
        filename = str(f"filename_{date}" + ".jpg")
        cv2.imwrite(filename, img) 

    def add_extracted_text(self, i, j, h, w, img, text):
        n = i * h
        m = j * w

        img = self.add_text_on_image(img, n, m, text, scale=0.5)
        
        return i * h, j * w, img

    def seperator(self):
        if self.optimization:
            for number in self.n_images:
                h = int(self.image_height / number)
                w = int(self.image_width / number)

                for i in range(number):
                    for j in range(int(number / 2), number):
                        self.add_extracted_img(i, j, w, h) 
        else:
            for number in self.n_images:
                h = int(self.image_height / number)
                w = int(self.image_width / number)

                for i in range(number):
                    for j in range(number):
                        self.add_extracted_img(i, j, w, h)
                

    def result_coloring(self, results):
        it = 0
        imgs = [self.img.copy() for _ in range(len(self.n_images))]

        if self.optimization:
            for img, number in zip(imgs, self.n_images):
                h = int(self.image_height / number)
                w = int(self.image_width / number)
                
                x, y = [], []

                for i in range(number):
                    for j in range(int(number / 2), number):
                        text = class_labels[results[it]] 

                        x_, y_, img = self.add_extracted_text(i, j, h, w, img, text)
                        x.append(x_)
                        y.append(y_)

                        it += 1
        else:
            for img, number in zip(imgs, self.n_images):
                h = int(self.image_height / number)
                w = int(self.image_width / number)
                
                x, y = [], []

                for i in range(number):
                    for j in range(number):
                        text = class_labels[results[it]] 

                        x_, y_, img= self.add_extracted_text(i, j, h, w, img, text)
                        x.append(x_)
                        y.append(y_)

                        it += 1
                        
        for s in x:
            img[s, :, :] = [255, 0, 0]
        
        for s in y:
            img[:, s, :] = [255, 0, 0]

        if self.visualization:
            self.plot_result(imgs)    

        return imgs

    def plot_result(self, imgs):
        image = None
        for i in range(len(imgs)):
            if i == 0:
                image=imgs[i]
            else:
                image = np.hstack((image, imgs[i]))
        
        cv2.imshow("Seperation", image)


    def show_seperation(self):
        imgs = [self.img.copy() for _ in range(len(self.n_images))]

        for img, number in zip(imgs, self.n_images):
            h = int(self.image_height / number)
            w = int(self.image_width / number)
            
            x, y = [], []
            it = 0
            for i in range(number):
                for j in range(number):
                    n = i * h
                    m = j * w

                    img = self.add_text_on_image(img, n, m, it)

                    x.append(i * h)
                    y.append(j * w)

                    it += 1

            for s in x:
                img[s, :, :] = [255, 0, 0]
            
            for s in y:
                img[:, s, :] = [255, 0, 0]

            #date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p_%f")
            #filename = str(f"filename_{date}" + ".jpg")
            #cv2.imwrite(filename, self.img)

            plt.figure()
            plt.imshow(img)
            plt.waitforbuttonpress(10)
            plt.close()
            
    
    def add_text_on_image(self, img, x, y, text, shift=75, scale=1):
        h, w = img.shape[:2]

        if not x+shift > h or not  y + shift > w:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(text), (y + shift, x + shift), font, scale, (0, 0, 255), 2, cv2.LINE_AA)
        
        return img

    def refactor_images(self): #TODO
        # This code is only for n_images (2, 4)
        # Rearange most important sub images
        last_img = self.sub_images[-self.n_images[-1]:]
        self.sub_images = self.sub_images[:-self.n_images[-1]]

        self.sub_images[4:4] = last_img
    
    def extract_image(self, x, y, w, h):
        if isinstance(self.img, np.ndarray):
            #("max row: {}, max column: {}".format(x+h, y+w))
            # if y+w < self.image_height and x+h < self.image_width:
            return self.img[x:x+h, y:y+w, :]  
        else:
            return None

    def save_image(self):
        date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p_%f")
        filename = str(f"filename_{date}" + ".jpg")
        cv2.imwrite(r"images/" + filename, self.img)


    def save_seperated_image(self):
        if self.buffer:
            for _ in range(self.count):
                element = self.buffer.popleft()
            
                if isinstance(element, np.ndarray):
                    date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p_%f")
                    filename = str(r"seperated/" + f"filename_{date}" + ".jpg")
                
                    cv2.imwrite(filename, element)
            self.clear()

    def prioritize_segments(self, y_labels):
        prior = np.vstack((y_labels, np.zeros(len(y_labels))))
        
        shift = 0
        for n in self.n_images:
            if self.optimization:
                p = pow(int(n / 2), 2)
                
                prior[1, shift+p:shift+p + p] = 1
                shift += n
            else:
                p = int(pow(n, 2 ) / 2)

                prior[1, shift:shift+p] = 1
                shift += n + n

        rem_nothing = prior[0, :] != 6
        prior = prior[:, rem_nothing]

        prior0 = prior[:, prior[1, :] == 0]
        prior1 = prior[:, prior[1, :] == 1]
        
        targetSignIdx0 = None
        targetSignIdx0 = None
        
        max0 = None
        max1 = None

        for i in range(6):
            m0 = (prior0[0, :] == i).sum()
            m1 = (prior1[0, :] == i).sum()
            print(i, m0, m1)
            if i == 0:
                max0 = m0
                max1 = m1

                targetSignIdx0 = i
                targetSignIdx1 = i
            
            if max0 < m0:
                max0 = m0
                targetSignIdx0 = i
    
            if max1 < m1:
                max1 = m1
                targetSignIdx1 = i

        print(targetSignIdx0, targetSignIdx1)

        if targetSignIdx0 > 1 and targetSignIdx1 > 1:
            traffic_signs = {
                0: class_labels[targetSignIdx0],
                1: class_labels[targetSignIdx1]
            }
        elif targetSignIdx0 == 0 and targetSignIdx1 > 1:
            traffic_signs = {
                1: class_labels[targetSignIdx1]
            }
        elif targetSignIdx0 > 1 and targetSignIdx1 == 0:
            traffic_signs = {
                0: class_labels[targetSignIdx0]
            }
        else:
            traffic_signs = dict()

        return traffic_signs