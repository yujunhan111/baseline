import torch

# 创建死亡和住院时长预测的权重
death_los_weights = {
    'death': torch.tensor([10.0]).cuda(),  # 死亡预测权重
    'los_gt_7': torch.tensor([2.0]).cuda()  # 住院时长大于7天预测权重
}