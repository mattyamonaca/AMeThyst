import gradio as gr
from PIL import Image, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tools import depth, sam, spectralresidual, brightness
from tools import depth
from utils import load_seg_model

import os


path = os.getcwd()
output_dir = f"{path}/output"
input_dir = f"{path}/input"
model_dir = f"{path}/models/sam"

load_seg_model(model_dir)


def create_heatmap_overlay(image, result, alpha=0.5, save_path='heatmap_overlay.png'):
    height, width = image.shape[:2]
    image = np.array(image, dtype=np.uint8)
    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(result, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.savefig(f'{path}/heatmap.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap = cv2.cvtColor(cv2.imread(f'{path}/heatmap.png'), cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (width, height))
    cv2.imwrite(f'{path}/heatmap.png', heatmap)
    overlay = cv2.addWeighted(color_image, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(save_path, overlay)
    return overlay


def plot_overlay(image, analysis_result):
    #image = image.convert('RGB')
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_overlay = create_heatmap_overlay(image_array, analysis_result, alpha=0.5, save_path='heatmap_overlay.png')
    return image_overlay

def composite(results):
    base = results[0][0] * results[0][1]
    for result in results[1:]:
        base = base + result[0] * result[1]
    result = base/len(results)
    return result

def analysis(
        image,
        sam_flag, depth_flag, spectralresidual_flag, brightness_flg,
        sam_heatmap_weight, sam_window_size, points_per_side, min_mask_region_area, pred_iou_thresh, stability_score_thresh, 
        depth_heatmap_weight, depth_window_size,
        spectralresidual_heatmap_weight,
        brightness_heatmap_weight, brightness_window_size,
):
    results = []
    if sam_flag == True:
        results.append((sam.process(model_dir, image, sam_window_size, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area), sam_heatmap_weight))
    if depth_flag == True:
        results.append((depth.process(image, depth_window_size), depth_heatmap_weight))
    if spectralresidual_flag == True:
        results.append((spectralresidual.process(image), spectralresidual_heatmap_weight))
    if brightness_flg == True:
        results.append((brightness.process(image, brightness_window_size), brightness_heatmap_weight))
        

    if len(results) > 1:
        result = composite(results)
    else:
        result = results[0][0]
    image_overlay = plot_overlay(image, result)
    return image_overlay


# Gradio UIの設定
enables = {}
sam_params = {}
with gr.Blocks() as demo:
    with gr.Row() as common:
        with gr.Column() as sub:
            input=gr.Image(type="pil")
            submit=gr.Button("submit")
            
            with gr.Accordion("Object Density") as sam_block:
                sam_flag = gr.Checkbox(False, label="enable")
                sam_heatmap_weight = gr.Slider(value = 1, minimum=0, maximum=1, step=0.05, label="heatmap_weight", randomize=True)
                sam_window_size = gr.Number(value=64, label="window_size")
                with gr.Accordion("SAM Parameter") as sam_param:
                    points_per_side = gr.Number(value=32, label="window_size")
                    min_mask_region_area=gr.Number(value=10, label="window_size")
                    pred_iou_thresh=gr.Slider(value = 0.8, minimum=0, maximum=1, step=0.05, label="pred_iou_thresh", randomize=True)
                    stability_score_thresh=gr.Slider(value = 0.8, minimum=0, maximum=1, step=0.05, label="stability_score_thresh", randomize=True)
                    
            with gr.Accordion("Subject Distance") as sam_block:
                depth_flag = gr.Checkbox(False, label="enable")
                depth_heatmap_weight = gr.Slider(value = 0, minimum=0, maximum=1, step=0.05, label="heatmap_weight", randomize=True)
                depth_window_size = gr.Number(value=64, label="window_size")

            with gr.Accordion("Spectral Residual") as sam_block:
                spectralresidual_flag = gr.Checkbox(False, label="enable")
                spectralresidual_heatmap_weight = gr.Slider(value = 0, minimum=0, maximum=1, step=0.05, label="heatmap_weight", randomize=True)

            with gr.Accordion("Brightness") as sam_block:
                brightness_flag = gr.Checkbox(False, label="enable")
                brightness_heatmap_weight = gr.Slider(value = 0, minimum=0, maximum=1, step=0.05, label="heatmap_weight", randomize=True)
                brightness_window_size = gr.Number(value=64, label="window_size")
                

        output=gr.Image(type="pil")
        submit.click(
            fn=analysis, inputs=[
                    input,
                    sam_flag, depth_flag, spectralresidual_flag, brightness_flag,
                    sam_heatmap_weight, sam_window_size, points_per_side, min_mask_region_area, pred_iou_thresh, stability_score_thresh, 
                    depth_heatmap_weight, depth_window_size,
                    spectralresidual_heatmap_weight,
                    brightness_heatmap_weight, brightness_window_size
                ], outputs=output)
# インターフェースの起動
demo.launch(share=True)
