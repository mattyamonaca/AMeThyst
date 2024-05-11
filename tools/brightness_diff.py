import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def convert_to_hsv_and_extract_brightness(image):
    """ 画像をHSVに変換し、明度（Vチャネル）を抽出 """
    image_np = np.array(image)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    brightness = hsv_image[:, :, 2]  # Vチャネルの抽出
    return brightness

def brightness_contrast_analysis(brightness, window_size=64):
    """ 指定されたウィンドウサイズでの明度の最大値と最小値の差を計算 """
    height, width = brightness.shape
    result = np.zeros((height, width))

    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            window = brightness[i:i + window_size, j:j + window_size]
            max_brightness = np.max(window)
            min_brightness = np.min(window)
            contrast = max_brightness - min_brightness
            result[i:i + window_size, j:j + window_size] += contrast

    return result

def create_heatmap(result):
    """ 結果をヒートマップとして表示 """
    plt.imshow(result, cmap='hot')
    plt.axis('off')
    plt.show()

def process(image, window_size=64):
    """ 画像処理の主要な関数 """
    brightness = convert_to_hsv_and_extract_brightness(image)
    contrast_result = brightness_contrast_analysis(brightness, window_size)
    return contrast_result
