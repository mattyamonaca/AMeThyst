
import os
import requests
from tqdm import tqdm


def load_seg_model(model_dir):
  folder = model_dir
  file_name = 'sam_vit_h_4b8939.pth'
  url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)