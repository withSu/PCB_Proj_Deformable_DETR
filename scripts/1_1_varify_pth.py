import torch

checkpoint_path = '/home/a/A_2024_selfcode/PCB_proj_DETR/0_output_Backup/checkpoint_099.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(checkpoint.keys())  # 체크포인트 키 확인

