import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import torch

# def get_history_codes(visits, current_time,index_set):
#     """历史codes收集函数"""
#     history_codes = []
#     for visit in visits:
#         # 先检查visit的第一个时间
#         first_code_time = visit[0][1]
#         if first_code_time >= current_time:
#             break
#         # 收集这个visit中的合适codes
#         for code_idx, code_time in visit:
#             if code_time >= current_time:
#                 break
#             if code_idx in index_set:  # 只保留有效的code
#                 history_codes.append((code_idx, code_time))
#     return history_codes if history_codes else None

def convert_to_serializable(obj):
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def print_memory_stats(prefix=""):
    print(f"\n{prefix} Memory Stats:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")

def force_memory_cleanup():
    """强制清理内存"""
    gc.collect()
    torch.cuda.empty_cache()



def get_history_codes(visits, current_time, index_set):
    """历史codes收集函数，返回分离后的code和time"""
    codes = []
    times = []

    for visit in visits:
        first_code_time = visit[0][1]
        if first_code_time >= current_time:
            break
        for code_idx, code_time in visit:
            if code_time >= current_time:
                break
            if code_idx in index_set and not pd.isna(code_time):
                codes.append(code_idx)
                times.append(code_time)

    return [codes, times] if codes else None


def check_patients_data(batch):
    print("\n=== Batch Structure Analysis ===")
    batch_size = len(batch['accumulated_codes'])
    print(f"\nBatch size: {batch_size}")

    # 检查每个病人的数据
    for patient_idx in range(batch_size):
        print(f"\n=== Patient {patient_idx} ===")

        # 检查demographic数据
        print("\nDemographic tensor:", batch['demographic'][patient_idx].shape)
        print("Demographic sample:", batch['demographic'][patient_idx])

        # 检查累积的codes和times
        acc_codes = batch['accumulated_codes'][patient_idx]
        acc_times = batch['accumulated_times'][patient_idx]
        print(f"\nAccumulated codes length: {len(acc_codes)}")
        print(f"Accumulated times length: {len(acc_times)}")
        print("Sample accumulated codes:", acc_codes[:5] if acc_codes else "Empty")
        print("Sample accumulated times:", acc_times[:5] if acc_times else "Empty")

        # 检查visit结束索引
        visit_indices = batch['visit_end_indices'][patient_idx]
        print("\nVisit end indices:", visit_indices)
        print("Number of visits:", len(visit_indices))

        # 验证每个visit的长度是否正确
        for i in range(len(visit_indices)):
            start_idx = 0 if i == 0 else visit_indices[i - 1]
            end_idx = visit_indices[i]
            print(f"Visit {i} length: {end_idx - start_idx}")

        # 检查真实时间点
        true_times = batch['true_last_visit_end_times'][patient_idx]
        print(f"\nTrue time points length: {len(true_times)}")
        print("Sample true times:", true_times if true_times else "Empty")


        # 检查真实的下一个codes
        next_codes = batch['true_next_codes'][patient_idx]
        print(f"\nTrue next codes length: {len(next_codes)}")
        print("Sample next codes:", next_codes if next_codes else "Empty")
        print("================================")
        print("Sample next codes(last):", next_codes[-1] )

        all_visit_end = batch['all_visit_end'][patient_idx]
        print("\nAll visit end time:", all_visit_end)

        # 检查采样点数据
        sampled_times = batch['sampled_time_points'][patient_idx]
        sampled_indices = batch['sampled_end_indices'][patient_idx]
        print(f"\nNumber of sampled points: {len(sampled_times)}")
        print("Sample sampled times:", sampled_times if sampled_times else "Empty")
        print("Sample sampled indices:", sampled_indices if sampled_indices else "Empty")


        # 数据一致性检查
        print("\n=== Consistency Checks ===")
        print(f"Times and codes match: {len(acc_times) == len(acc_codes)}")
        print(f"Sampled times and indices match: {len(sampled_times) == len(sampled_indices)}")

        # 检查时间顺序
        if acc_times:
            is_sorted = all(acc_times[i] <= acc_times[i + 1] for i in range(len(acc_times) - 1))
            print(f"Times are in order: {is_sorted}")

        # 检查索引是否有效
        if sampled_indices:
            valid_indices = all(idx <= len(acc_codes) for idx in sampled_indices)
            print(f"All sample indices are valid: {valid_indices}")

        print("\n" + "=" * 50)


def check_patients_data_diease(batch):
    """
    打印批次数据的详细信息，包括batch size检查
    """
    print("\n====== 批次数据检查 ======")

    # 检查batch size
    demographic = batch['demographic']
    disease_data = batch['disease_data']
    death_data = batch['death_labels']
    demo_batch_size = len(demographic)
    disease_batch_size = len(disease_data)
    death_batch_size = len(death_data)

    print("\n[Batch Size 信息]")
    print(f"Demographic batch size: {demo_batch_size}")
    print(f"Disease data batch size: {disease_batch_size}")
    print(f"Death data batch size: {death_batch_size}")

    if not (demo_batch_size == disease_batch_size == death_batch_size):
        print(f"警告: Demographic、Disease和Death数据的batch size不一致!")
        return

    print(f"\n开始检查 {demo_batch_size} 个样本的数据...")

    # 1. 打印人口统计数据信息
    print("\n[人口统计数据]")
    print(f"Shape: {demographic.shape}")
    print(f"数据类型: {demographic.dtype}")
    print(f"所在设备: {demographic.device}")

    for i in range(demo_batch_size):
        print(f"\n样本 {i + 1} 的demographic数据:")
        print(demographic[i])

    # 2. 打印疾病数据
    print("\n[疾病数据]")

    # 疾病统计计数器
    disease_stats = {
        'positive': 0,
        'negative': 0,
        'undefined': 0
    }

    # 检查每个样本的疾病数据
    for sample_idx in range(disease_batch_size):
        sample_diseases = disease_data[sample_idx]
        print(f"\n样本 {sample_idx + 1} 的疾病信息:")

        for disease_name, disease_info in sample_diseases.items():
            label = disease_info['label']
            if label == 0.95:
                label_type = "阳性"
                disease_stats['positive'] += 1
            elif label == 0.05:
                label_type = "阴性"
                disease_stats['negative'] += 1
            else:
                label_type = "未定义"
                disease_stats['undefined'] += 1

            print(f"\n疾病: {disease_name}")
            print(f"标签: {label_type} ({label})")
            print(f"事件时间: {disease_info['event_time']}")

            history_codes = disease_info['history_codes']
            history_times = disease_info['history_times']

            print(f"历史代码数量: {len(history_codes)}")
            print(f"历史时间数量: {len(history_times)}")

            if len(history_codes) > 0:
                print("历史代码后5位:", history_codes[-5:])
                print("历史时间后5位:", history_times[-5:])

    # 3. 打印死亡数据
    print("\n[死亡数据]")

    # 死亡统计计数器
    death_stats = {
        'positive': 0,
        'negative': 0,
        'undefined': 0
    }

    # 检查每个样本的死亡数据
    for sample_idx in range(death_batch_size):
        sample_death = death_data[sample_idx]
        print(f"\n样本 {sample_idx + 1} 的死亡信息:")

        for death_type, death_info in sample_death.items():
            label = death_info['label']
            if label == 0.95:
                label_type = "阳性"
                death_stats['positive'] += 1
            elif label == 0.05:
                label_type = "阴性"
                death_stats['negative'] += 1
            else:
                label_type = "未定义"
                death_stats['undefined'] += 1

            print(f"\n死亡类型: {death_type}")
            print(f"标签: {label_type} ({label})")
            print(f"事件时间: {death_info['event_time']}")

            history_codes = death_info['history_codes']
            history_times = death_info['history_times']

            print(f"历史代码数量: {len(history_codes)}")
            print(f"历史时间数量: {len(history_times)}")

            if len(history_codes) > 0:
                print("历史代码前5位:", history_codes[:5])
                print("历史时间前5位:", history_times[:5])

    # 打印总体统计
    print("\n[总体统计]")
    print(f"总样本数: {demo_batch_size}")
    print("\n疾病统计:")
    print(f"阳性病例数: {disease_stats['positive']}")
    print(f"阴性病例数: {disease_stats['negative']}")
    print(f"未定义病例数: {disease_stats['undefined']}")
    print("\n死亡统计:")
    print(f"阳性病例数: {death_stats['positive']}")
    print(f"阴性病例数: {death_stats['negative']}")
    print(f"未定义病例数: {death_stats['undefined']}")
    print("\n====== 数据检查完成 ======\n")


def plot_all_roc_curves(disease_metrics, death_metrics):
    """
    绘制 ROC 曲线，跳过无效的指标

    Args:
        disease_metrics (dict): 疾病预测的指标
        death_metrics (dict): 死亡预测的指标
    """
    from sklearn.metrics import roc_curve
    import numpy as np


    plt.figure(figsize=(12, 8))  # 增大图形尺寸以适应更多图例

    # 处理所有疾病的预测结果
    for disease_name, metrics in disease_metrics.items():
        y_true = metrics['y_true']
        y_score = metrics['y_score']

        if len(y_true) > 0 and len(np.unique(y_true)) > 1:  # 确保有数据且不是单一类别
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr,
                         label=f'{disease_name} (AUC = {roc_auc:.3f})',
                         linewidth=2)
            except Exception as e:
                plt.plot([], [],
                         label=f'{disease_name} (AUC = nan)',
                         linewidth=0)  # 空线条，只显示在图例中
        else:
            # 对于无效数据，只添加图例条目
            plt.plot([], [],
                     label=f'{disease_name} (AUC = nan)',
                     linewidth=0)

    # 处理所有死亡类型的预测结果
    for death_type, metrics in death_metrics.items():
        y_true = metrics['y_true']
        y_score = metrics['y_score']

        if len(y_true) > 0 and len(np.unique(y_true)) > 1:  # 确保有数据且不是单一类别
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr,
                         label=f'{death_type} (AUC = {roc_auc:.3f})',
                         linewidth=2)
            except Exception as e:
                plt.plot([], [],
                         label=f'{death_type} (AUC = nan)',
                         linewidth=0)
        else:
            # 对于无效数据，只添加图例条目
            plt.plot([], [],
                     label=f'{death_type} (AUC = nan)',
                     linewidth=0)

    # 添加对角线
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')

    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Disease and Death Prediction')

    # 调整图例位置和样式
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize=8, frameon=True,
               fancybox=True, shadow=True)

    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像，确保留出足够空间给图例
    plt.tight_layout()
    save_path = 'combined_roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curves have been saved to: {save_path}")
def get_current_demographic(demographic, current_time):
    """
    根据当前时间更新人口统计特征中的年龄
    Args:
        demographic: 原始人口统计特征 [1, dim]
        current_time: 当前相对时间点(log1p(hours/24/7))
    Returns:
        demographic_current: 更新后的特征
    """
    demographic_current = demographic.clone()
    current_years = np.expm1(current_time) * 7 * 24 / (365 * 24)
    demographic_current[0, 1] = torch.log1p(torch.expm1(demographic_current[0, 1]) + current_years)
    return demographic_current