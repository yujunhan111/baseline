import torch


def custom_collate_full(batch):
    """
    包含疾病预测、死亡预测和LOS预测的自定义collate函数

    Args:
        batch (list): 包含字典的列表，每个字典有'demographic', 'disease_data', 'death_los_data'

    Returns:
        dict: 处理后的batch，包含:
            - 'demographic': 人口统计学特征张量
            - 'disease_data': 疾病数据字典列表
            - 'death_los_data': 死亡和LOS数据字典列表
    """
    return {
        'demographic': torch.stack([item['demographic'] for item in batch]),
        'disease_data': [item['disease_data'] for item in batch],
        'death_los_data': [item['death_los_data'] for item in batch]
    }