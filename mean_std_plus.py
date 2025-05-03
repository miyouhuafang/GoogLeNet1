import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # 用于显示进度条

# 原始文件夹路径
folder_path = r'D:\pycharm\GoodLeNet\cat and dog\data\train'


def calculate_mean_variance(folder_path):
    """
    计算图像数据集的逐通道均值和方差（基于全部训练数据）

    参数:
        folder_path (str): 包含训练图像的文件夹路径

    返回:
        mean (np.array): RGB三通道均值
        variance (np.array): RGB三通道方差
    """
    # 初始化累加器
    sum_pixels = np.zeros(3)  # 存储RGB三通道的像素值总和
    sum_squares = np.zeros(3)  # 存储RGB三通道的像素值平方总和
    num_pixels = 0  # 总像素数（所有图像的 宽×高 之和）

    # 第一次遍历：计算像素总和和平方总和
    print("正在计算均值和方差...")

    # 遍历文件夹中的所有图像
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):  # 使用tqdm显示进度条
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # 支持常见格式
                image_path = os.path.join(root, file)
                try:
                    # 使用with语句确保文件正确关闭
                    with Image.open(image_path) as img:
                        # 确保图像为RGB格式（处理灰度图或其他模式）
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # 转换为numpy数组并归一化到[0,1]
                        img_array = np.array(img) / 255.0

                        # 获取图像尺寸
                        h, w = img_array.shape[:2]

                        # 累加总像素数（每个像素的三个通道独立计算）
                        num_pixels += h * w

                        # 计算各通道总和（axis=(0,1)表示对高度和宽度求和）
                        sum_pixels += np.sum(img_array, axis=(0, 1))

                        # 计算各通道平方和
                        sum_squares += np.sum(img_array ** 2, axis=(0, 1))
                except Exception as e:
                    print(f"处理图像 {image_path} 时出错: {str(e)}")
                    continue  # 跳过问题图像

    # 计算均值
    mean = sum_pixels / num_pixels

    # 计算方差 = E[X²] - (E[X])²
    variance = (sum_squares / num_pixels) - mean ** 2


    return mean, variance


# 调用函数
mean, variance = calculate_mean_variance(folder_path)
std_dev = np.sqrt(variance)
print(f"均值 (RGB): {mean}")
print(f"方差 (RGB): {std_dev}")