import os
import json
import random
from shutil import copy2

def split_and_convert_to_coco(json_dir, image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, image_width=3904, image_height=3904):
    def initialize_coco():
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "component"
                }
            ]
        }

    # 결과 디렉토리 설정
    train_dir = os.path.join(output_dir, "train_images")
    val_dir = os.path.join(output_dir, "val_images")
    test_dir = os.path.join(output_dir, "test_images")

    train_annotations_file = os.path.join(output_dir, "annotations", "train.json")
    val_annotations_file = os.path.join(output_dir, "annotations", "val.json")
    test_annotations_file = os.path.join(output_dir, "annotations", "test.json")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # JSON 파일 목록 가져오기
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    random.shuffle(json_files)
    
    train_split = int(len(json_files) * train_ratio)
    val_split = int(len(json_files) * (train_ratio + val_ratio))

    train_files = json_files[:train_split]
    val_files = json_files[train_split:val_split]
    test_files = json_files[val_split:]

    # COCO 데이터 초기화
    train_coco = initialize_coco()
    val_coco = initialize_coco()
    test_coco = initialize_coco()

    # ID 초기화
    image_id = 1
    annotation_id = 1

    # JSON 파일 처리
    def process_files(file_list, target_dir, coco_format):
        nonlocal image_id, annotation_id
        for json_filename in file_list:
            json_filepath = os.path.join(json_dir, json_filename)

            # JSON 파일 읽기
            with open(json_filepath, 'r') as f:
                input_json = json.load(f)

            # 이미지 파일 확인
            image_filename_base = json_filename.replace('.json', '')
            image_filename = None
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = os.path.join(image_dir, image_filename_base + ext)
                if os.path.isfile(candidate):
                    image_filename = image_filename_base + ext
                    break

            if not image_filename:
                print(f"Warning: No matching image found for {json_filename}")
                continue

            # 이미지 복사
            copy2(os.path.join(image_dir, image_filename), target_dir)

            # 이미지 정보 추가
            coco_format["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height
            })

            # 어노테이션 추가
            for shape in input_json["shapes"]:
                points = shape["points"]
                x1, y1 = points[0]
                x2, y2 = points[1]
                bbox = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    # 학습 데이터 처리
    process_files(train_files, train_dir, train_coco)
    # 검증 데이터 처리
    process_files(val_files, val_dir, val_coco)
    # 테스트 데이터 처리
    process_files(test_files, test_dir, test_coco)

    # COCO JSON 저장
    with open(train_annotations_file, 'w') as f:
        json.dump(train_coco, f, indent=4)
    with open(val_annotations_file, 'w') as f:
        json.dump(val_coco, f, indent=4)
    with open(test_annotations_file, 'w') as f:
        json.dump(test_coco, f, indent=4)

    print(f"Train images: {len(train_files)}, Val images: {len(val_files)}, Test images: {len(test_files)}")
    print("COCO format JSON files created!")

# 실행
json_directory = '/home/user/Desktop/Vision-team/KBS/PCB_Deformable-DETR/raw_datasets/2_raw_json'  # JSON 파일 경로
image_directory = '/home/user/Desktop/Vision-team/KBS/PCB_Deformable-DETR/1_images'  # 이미지 파일 경로
output_directory = '/home/user/Desktop/Vision-team/KBS/PCB_Deformable-DETR/datasets'  # 결과 저장 경로

split_and_convert_to_coco(json_directory, image_directory, output_directory, train_ratio=0.7, val_ratio=0.2)