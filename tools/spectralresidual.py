import cv2
import numpy as np

def process(image):
    image_array = np.array(image)
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # アルゴリズムの設定
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # サリエンシーディテクション
    bool, map = saliency.computeSaliency(image)
    i_saliency = (map * 255).astype("uint8")
    return i_saliency
