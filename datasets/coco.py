# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset for Deformable DETR, returns image_id for evaluation.
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from datasets.transforms import Compose, RandomHorizontalFlip, RandomSelect, RandomResize, ToTensor, Normalize


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        print(f"📂 Initializing CocoDetection with ann_file: {ann_file}")
        if not Path(ann_file).exists():
            raise ValueError(f"❌ Annotation file {ann_file} does not exist!")

        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        # 여러 개의 RLE가 합쳐져 있을 경우, any(dim=2)로 통합
        mask = mask.any(dim=2)
        masks.append(mask)

    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    """
    단일 클래스(예: category_id=1)만 사용할 경우:
      - cid == 1인 객체만 남기고, 나머지 category_id는 제거.
      - 모델에 입력할 라벨은 전부 0으로 통일 (foreground).
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = torch.tensor([target["image_id"]], dtype=torch.int64)
        anno = target["annotations"]

        # (1) crowd가 아닌(iscrowd=0) 객체만 남기기
        # (2) category_id=1인 객체만 남기기
        filtered_anno = []
        for obj in anno:
            if obj.get('iscrowd', 0) == 1:
                continue
            # 🔴 기존 코드: if obj["category_id"] == 0:
            # 🔴 수정: category_id=1인 객체를 필터링
            if obj["category_id"] == 1:
                filtered_anno.append(obj)

        boxes = [obj["bbox"] for obj in filtered_anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # xywh -> xyxy
        boxes[:, 2:] += boxes[:, :2]
        # clamp
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 🔴 전부 label=0으로. (foreground)
        classes = torch.zeros((len(filtered_anno),), dtype=torch.int64)

        # (필요하다면 masks를 구함)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in filtered_anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        else:
            masks = None

        # width/height가 0 이하인 박스 제거
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]

        area = torch.as_tensor([obj["area"] for obj in filtered_anno], dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get("iscrowd", 0) for obj in filtered_anno], dtype=torch.int64)
        area = area[keep]
        iscrowd = iscrowd[keep]

        # 최종 target 구성
        new_target = {}
        new_target["boxes"] = boxes
        new_target["labels"] = classes  # 전부 0
        if masks is not None:
            new_target["masks"] = masks
        new_target["image_id"] = image_id
        new_target["area"] = area
        new_target["iscrowd"] = iscrowd
        new_target["orig_size"] = torch.as_tensor([int(h), int(w)], dtype=torch.int64)
        new_target["size"] = torch.as_tensor([int(h), int(w)], dtype=torch.int64)

        return image, new_target


def make_coco_transforms(image_set):
    """
    COCO 학습 시 일반적으로 사용하는 변환 + Normalize
    """
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            RandomSelect(
                RandomResize([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], max_size=1333),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomResize([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    """
    COCO dataset builder.
    여기서 'train', 'val', 'test' 세 가지 모드에 맞춰서 annotation 파일 경로와 이미지 폴더를 설정한다.
    """
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    PATHS = {
        "train": (root / "train_images", root / "annotations" / "train.json"),
        "val":   (root / "val_images",   root / "annotations" / "val.json"),
        "test":  (root / "test_images",  root / "annotations" / "test.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    if not ann_file.exists():
        raise ValueError(f"❌ Annotation file {ann_file} does not exist!")

    print(f"✅ Loading COCO dataset from: {ann_file}")

    # transforms
    transforms = make_coco_transforms(image_set)

    # CocoDetection 생성
    dataset = CocoDetection(
        img_folder, ann_file,
        transforms=transforms,
        return_masks=args.masks
    )

    return dataset
