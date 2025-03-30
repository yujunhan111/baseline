import torch


def custom_collate(batch):
    """
    自定义的collate函数，处理变长序列的批处理
    """
    return {
        'demographic': torch.stack([item['demographic'] for item in batch]),
        # 累积的历史数据
        'accumulated_codes': [item['accumulated_codes'] for item in batch],
        'accumulated_times': [item['accumulated_times'] for item in batch],
        'visit_end_indices': [item['visit_end_indices'] for item in batch],
        # 真实visit数据
        'true_last_visit_end_times': [item['true_last_visit_end_times'] for item in batch],
        'true_next_codes': [item['true_next_codes'] for item in batch],
        # 采样点数据
        'sampled_time_points': [item['sampled_time_points'] for item in batch],
        'sampled_end_indices': [item['sampled_end_indices'] for item in batch],
        # 最后访问时间评估数据
        'last_visit_eval_time': [item['last_visit_eval_time'] for item in batch],
        # 所有visit的最后时间
        'all_visit_end': [item['all_visit_end'] for item in batch]
    }