import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む
image_path = 'input/test1.png'  # 画像ファイルのパスを指定
image = cv2.imread(image_path)

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 画像の勾配（グラデーション）を計算
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X方向の勾配
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y方向の勾配

# 勾配の大きさを計算
magnitude = np.sqrt(sobelx**2 + sobely**2)

# ヒートマップを生成
plt.figure(figsize=(10, 10))
plt.imshow(magnitude, cmap='coolwarm')  # 青から赤のカラースケールを使用
plt.colorbar(label="Detail Level")
plt.axis('off')  # 軸を非表示
plt.show()