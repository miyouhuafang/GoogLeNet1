import os
import numpy as np
from PIL import Image

folder_path = r'D:\pycharm\GoodLeNet\cat and dog\data\cats_and_dogs\train'

total_pixels = 0
sum_normalized_pixel_values = np.zeros(3) #RGB图像

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            image_array = np.array(image)

            #归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            total_pixels = total_pixels + normalized_image_array.size
            sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0,1))

#计算均值
mean=sum_normalized_pixel_values/total_pixels

sum_square_diff = np.zeros(3)
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            image_array = np.array(image)
            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            try:
                diff = (normalized_image_array - mean)**2
                sum_square_diff += np.sum(diff, axis=(0,1))
            except:
                print("catch the error")

#计算方差
variance = sum_square_diff/total_pixels
print("M:",mean)
print("V:",variance)