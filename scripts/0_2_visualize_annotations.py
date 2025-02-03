import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def visualize_coco_annotations(image_dir, json_file, num_images=5):
    # COCO 어노테이션 JSON 파일 로드
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 이미지 ID와 파일 이름 매핑
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 어노테이션 데이터 로드
    annotations = coco_data['annotations']
    
    # 이미지와 어노테이션 매칭
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
    
    # 시각화할 이미지 선택
    for img_id, img_file in list(images.items())[:num_images]:
        img_path = Path(image_dir) / img_file
        if not img_path.exists():
            print(f"이미지 {img_path}를 찾을 수 없습니다.")
            continue
        
        # 이미지 로드
        img = Image.open(img_path)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # 어노테이션 그리기
        if img_id in ann_by_image:
            for ann in ann_by_image[img_id]:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x, y, str(ann['category_id']), color='blue', fontsize=12)
        
        plt.axis('off')
        plt.show()

# 이미지 디렉토리와 어노테이션 파일 경로
image_dir = '/home/user/Desktop/Vision-team/KBS/PCB_Deformable-DETR/1_images'
json_file = '/home/user/Desktop/Vision-team/KBS/PCB_Deformable-DETR/datasets'

# 시각화 실행
visualize_coco_annotations(image_dir, json_file, num_images=5)
