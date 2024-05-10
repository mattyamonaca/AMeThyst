import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def load_and_process_image(image):
    image = image.convert('RGB')
    transform = Compose([
        Resize((384, 384)),  # MiDaS モデルの入力サイズに合わせてリサイズ
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def generate_depth_map(image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Torch HubからMiDaSモデルをロード
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model.to(device).eval()

    image_tensor = image_tensor.to(device)  # デバイスへの移動
    with torch.no_grad():
        depth_map = model(image_tensor)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=image_tensor.shape[2:],
            mode='bilinear',
            align_corners=True
        ).squeeze().cpu().numpy()

    return depth_map

def depth_analysis(depth_map, window_size = 64):
    height, width = depth_map.shape
    result = np.zeros((height, width))

    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            window = depth_map[i:i + window_size, j:j + window_size]
            average_depth = np.mean(window)
            result[i:i + window_size, j:j + window_size] += average_depth / (window_size ** 2)
    
    return result

def create_heatmap_overlay(image, result, alpha=0.5, save_path='heatmap_overlay.png'):
    height, width = image.shape[:2]
    image = np.array(image, dtype=np.uint8)
    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(result, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.savefig('heatmap.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap = cv2.imread('heatmap.png')
    heatmap = cv2.resize(heatmap, (width, height))
    cv2.imwrite('heatmap.png', heatmap)
    overlay = cv2.addWeighted(color_image, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(save_path, overlay)
    return overlay

def process(image, window_size):
    image_tensor = load_and_process_image(image)
    depth_map = generate_depth_map(image_tensor)
    depth_analysis_result = depth_analysis(depth_map, window_size)

    depth_analysis_result = cv2.resize(depth_analysis_result, (image.width, image.height))

    return depth_analysis_result

# 実行例
# result_image = analysis('path_to_your_image.jpg')
# result_image.show()
