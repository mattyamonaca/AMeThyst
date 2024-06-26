import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def convert_to_hsv_and_extract_brightness(image):
    """ 画像をHSVに変換し、明度（Vチャネル）を抽出 """
    image_np = np.array(image)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    brightness = hsv_image[:, :, 2]  # Vチャネルの抽出
    return brightness

def brightness_analysis(brightness, window_size=64):
    """ 指定されたウィンドウサイズでの明度の平均を計算 """
    height, width = brightness.shape
    result = np.zeros((height, width))

    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            window = brightness[i:i + window_size, j:j + window_size]
            average_brightness = np.mean(window)
            result[i:i + window_size, j:j + window_size] += average_brightness / (window_size ** 2)

    return result


def process(image, window_size=64):
    """ 画像処理の主要な関数 """
    brightness = convert_to_hsv_and_extract_brightness(image)
    brightness_result = brightness_analysis(brightness, window_size)
    return brightness_result
