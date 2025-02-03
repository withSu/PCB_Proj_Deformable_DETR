# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


from datasets.coco import CocoDetection
from pycocotools.coco import COCO

def get_coco_api_from_dataset(dataset):
    """데이터셋에서 COCO API를 가져오는 함수"""

    # 1️⃣ dataset이 CocoDetection 인스턴스인지 확인
    if isinstance(dataset, CocoDetection):
        print("✅ dataset은 CocoDetection 타입입니다.")  # 디버깅용 출력
        return dataset.coco  # CocoDetection의 COCO 객체 반환

    # 2️⃣ dataset이 리스트거나 여러 개의 데이터셋을 포함하는 경우 첫 번째 항목을 가져옴
    if isinstance(dataset, list):
        print("✅ dataset이 리스트입니다. 첫 번째 요소를 사용합니다.")  # 디버깅용 출력
        return get_coco_api_from_dataset(dataset[0])

    # 3️⃣ dataset이 DataLoader일 가능성이 있음 → dataset.dataset 확인
    if hasattr(dataset, 'dataset'):
        print("✅ dataset은 DataLoader입니다. dataset.dataset을 사용합니다.")  # 디버깅용 출력
        return get_coco_api_from_dataset(dataset.dataset)

    # 4️⃣ 여기에 도달하면 dataset이 COCO 객체를 반환하지 못하는 것 → 오류 발생
    print("❌ dataset에서 COCO API를 찾을 수 없습니다. 타입:", type(dataset))
    return None


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
