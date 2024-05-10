import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# モデルの読み込み (ここではVGG16を使用)
model = models.vgg16(pretrained=True)
model.eval()

# 画像の読み込み
img_path = './input/test3.jpg'
img = Image.open(img_path)
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)

# 勾配の計算
img_tensor.requires_grad_()
output = model(img_tensor)
pred_idx = torch.argmax(output)

model.zero_grad()
output[0, pred_idx].backward()

# サリエンシーマップの生成
saliency_map = img_tensor.grad.abs().squeeze().numpy()
saliency_map = np.max(saliency_map, axis=0)

# サリエンシーマップの表示
plt.imshow(saliency_map, cmap='hot')
plt.axis('off')
plt.show()
