import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def load_and_process_image(image):
    # 画像を読み込む
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_array = np.array(image)
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # オリジナルのサイズを取得
    original_height, original_width, _ = image.shape

    
    # 横幅を256に固定し、縦幅を比率を保ってリサイズ
    aspect_ratio = original_height / original_width
    new_width = 256
    new_height = int(new_width * aspect_ratio)
    
    # リサイズ
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image

def visualize_clusters(image_cls):
    # クラスタインデックスの最大値を取得
    num_clusters = np.max(image_cls)
    # クラスタごとにランダムな色を生成
    colors = np.random.randint(0, 255, (num_clusters + 1, 3))
    # カラー画像を作成
    height, width = image_cls.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(1, num_clusters + 1):
        color_image[image_cls == i] = colors[i]
    # 画像を表示
    plt.imshow(color_image)
    plt.axis('off')
    plt.show()
    cv2.imwrite("seg_mask.png", color_image)

def create_heatmap_overlay(image, result, alpha=0.5, save_path='heatmap_overlay.png'):
    height, width = image.shape[:2]  # オリジナル画像の高さと幅を取得

    # データ型をチェック・変換
    image = np.array(image, dtype=np.uint8)

    # カラースケール画像を作成
    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # カラーマップでヒートマップを生成
    plt.imshow(result, cmap='jet', alpha=alpha)
    plt.axis('off')

    # ヒートマップを画像として保存
    plt.savefig('heatmap.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # ヒートマップを読み込み
    heatmap = cv2.cvtColor(cv2.imread('heatmap.png'), cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (width, height))

    # ヒートマップとオリジナル画像を合成
    overlay = cv2.addWeighted(color_image, 1-alpha, heatmap, alpha, 0)

    # オーバーレイを画像として保存
    cv2.imwrite(save_path, overlay)

    return overlay

def cluster_analysis(image, window_size = 64):
    height, width = image.shape[:2]
    result = np.zeros((height, width))
    cluster_count_sum = np.zeros((height, width))
    window_count = np.zeros((height, width))

    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            window = image[i:i + window_size, j:j + window_size]
            unique_clusters = np.unique(window)
            cluster_count = len(unique_clusters)

            cluster_count_sum[i:i + window_size, j:j + window_size] += cluster_count
            window_count[i:i + window_size, j:j + window_size] += 1

    result = cluster_count_sum / window_count
    return result

def create_image_from_segmentations(segmentations):
    height, width = segmentations[0].shape
    image = np.zeros((height, width), dtype=int)

    for cluster_index, seg in enumerate(segmentations):
        image[seg] = cluster_index + 1

    return image

def create_image_from_anns(anns):
    """
    'show_anns()'関数で表示されるセグメンテーションに基づいて、
    クラスタインデックスを割り振る画像を生成する。
    :param anns: セグメンテーション情報のリスト
    :return: クラスタインデックスで埋められた2D配列
    """
    if len(anns) == 0:
        return None

    # セグメンテーションをサイズ順にソート
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    height, width = sorted_anns[0]['segmentation'].shape

    # インデックスを保持するための配列を初期化
    image_cls = np.zeros((height, width), dtype=int)

    # 各クラスタに対してセグメンテーションマスクを適用
    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        image_cls[m] = idx + 1  # クラスタインデックスを1から始める

    return image_cls

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return img

# 入力画像の読み込み
def process(
        image, window_size ,
        points_per_side,
        pred_iou_thresh,
        stability_score_thresh,
        min_mask_region_area,
        ):
    print(min_mask_region_area)
    width = image.width
    height = image.height
    image = load_and_process_image(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_region_area,
    )

    masks = mask_generator.generate(image)
    segs = [mask["segmentation"] for mask in masks]

    # クラスタ情報の配列を生成
    image_cls = create_image_from_anns(masks)
    #visualize_clusters(image_cls)
    result = cluster_analysis(image_cls, window_size)
    result = cv2.resize(result, (width, height))

    return result

