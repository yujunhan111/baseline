import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set
import gc


def thinning_predict_next_time(model, accumulated_codes, accumulated_times, current_time,
                               demographic, max_window, max_iter=10000):
    """
    使用Thinning Algorithm预测下一次访问时间

    Args:
        model: EHR模型实例
        accumulated_codes: 历史诊断代码序列
        accumulated_times: 历史访问时间序列
        current_time: 当前时间点
        demographic: 人口统计学特征
        max_window: 最大预测时间窗口
        max_iter: 最大迭代次数

    Returns:
        predicted_time: 预测的下一次访问时间
    """
    with torch.no_grad():
        # 1. 估计强度函数的上界
        sample_times = np.linspace(current_time + 0.00001, max_window, 50)
        max_intensity = -float('inf')

        for t in sample_times:
            # 获取历史code的嵌入和时间
            hist_embeddings, hist_times = model.get_code_embeddings(
                accumulated_codes,
                accumulated_times
            )

            # 获取当前时间点的表示
            event_representations = model.attention(
                hist_embeddings,
                hist_embeddings,
                hist_times
            )
            repr = model.history_repr(event_representations, hist_times, t)

            if repr is not None:
                intensity = model.intensity_net(demographic, repr.unsqueeze(0))
                max_intensity = max(max_intensity, intensity.item())
                del repr
                del intensity
                torch.cuda.empty_cache()

        # 增加安全边界
        M = max_intensity * 20

        # 2. Thinning Algorithm主循环
        for _ in range(max_iter):
            # 生成候选时间
            u1 = np.random.uniform(0, 1)
            tau = -np.log(u1) / M
            candidate_time = current_time + tau

            # if candidate_time > max_window:
            #     continue

            # 获取候选时间点的表示
            hist_embeddings, hist_times = model.get_code_embeddings(
                accumulated_codes,
                accumulated_times
            )
            event_representations = model.attention(
                hist_embeddings,
                hist_embeddings,
                hist_times
            )
            repr = model.history_repr(event_representations, hist_times, candidate_time)

            if repr is not None:
                intensity = model.intensity_net(demographic, repr.unsqueeze(0))
                # 计算接受概率
                acceptance_prob = intensity.item() / M
                u2 = np.random.uniform(0, 1)
                if u2 <= acceptance_prob:
                    return candidate_time
                del repr
                del intensity
                torch.cuda.empty_cache()

        # 如果达到最大迭代次数，返回期望时间
        return current_time + 1 / max_intensity
def compute_prediction_metrics(predictions: torch.Tensor, true_codes: List[int], k_values: List[int]) -> Dict[
    str, float]:
    """
    计算不同k值的预测指标

    Args:
        predictions: [1, code_vocab_size] 预测概率分布
        true_codes: 实际的诊断代码列表
        k_values: 要计算的k值列表

    Returns:
        包含所有k值的precision, recall, f1指标的字典
    """
    metrics = {}
    pred_probs = predictions.detach().cpu().numpy()
    true_codes_set = set(true_codes)

    for k in k_values:
        top_k_indices = np.argpartition(pred_probs[0], -k)[-k:]
        # 转换为实际的code index (因为true_codes从1开始)
        pred_codes = set(idx + 1 for idx in top_k_indices)
        # if(k == 5):
        #     print("pred_codes",pred_codes)
        true_positives = len(pred_codes.intersection(true_codes_set))
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(true_codes_set) if true_codes_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'f1@{k}'] = f1

    return metrics


def evaluate_model(model: torch.nn.Module,
                   val_loader: torch.utils.data.DataLoader,
                   k_values: List[int]) -> Dict[str, float]:
    """
    评估模型在所有病人最后一次访问的预测性能，包括诊断代码和下次就诊时间的预测

    Returns:
        Dict: 包含代码预测指标的平均值和原始的时间预测误差数据
    """
    model.eval()
    all_metrics = defaultdict(list)
    time_errors = []
    relative_errors = []

    with torch.no_grad():
        for batch in val_loader:
            try:
                for i in range(len(batch['last_visit_eval_time'])):
                    # 检查是否有有效的最后访问评估时间
                    eval_time = batch['last_visit_eval_time'][i]
                    demographic = batch['demographic'][i].unsqueeze(0).cuda()
                    # 获取最后一次和倒数第二次访问的时间
                    current_time = eval_time['second_last_visit_end']  # 倒数第二次就诊结束时间
                    true_next_time = eval_time['last_visit_start']  # 最后一次就诊开始时间
                    # 获取最后一次访问的诊断代码
                    last_visit_index = batch['visit_end_indices'][i][-1]
                    true_next_codes = batch['true_next_codes'][i][-1]

                    # 准备历史数据（不包括最后一次访问）
                    accumulated_codes = batch['accumulated_codes'][i][:last_visit_index]
                    accumulated_times = batch['accumulated_times'][i][:last_visit_index]
                    # print("accumulated_codes_len",len(accumulated_codes))
                    # print("accumulated_times_len",len(accumulated_times))
                    # print("accumulated_codes",accumulated_codes)
                    # print("accumulated_times",accumulated_times)
                    # print("true_next_codes_len",len(true_next_codes))
                    # print("true_next_codes",true_next_codes)
                    # 设置预测时间窗口
                    max_window = current_time + 2 * (true_next_time - current_time)

                    # 预测下次就诊时间
                    predicted_time = thinning_predict_next_time(
                        model=model,
                        accumulated_codes=accumulated_codes,
                        accumulated_times=accumulated_times,
                        current_time=current_time,
                        demographic=demographic,
                        max_window=max_window
                    )

                    # 计算时间预测误差
                    if predicted_time is not None:
                        predicted_time = np.expm1(predicted_time)
                        true_next_time_exp = np.expm1(true_next_time)
                        current_time_exp = np.expm1(current_time)

                        time_diff = abs(predicted_time - true_next_time_exp)
                        relative_error = time_diff / (true_next_time_exp - current_time_exp)

                        time_errors.append(time_diff)
                        relative_errors.append(relative_error)

                    # 获取历史表示用于预测诊断代码
                    hist_embeddings, hist_times = model.get_code_embeddings(
                        accumulated_codes,
                        accumulated_times
                    )

                    event_representations = model.attention(
                        hist_embeddings,
                        hist_embeddings,
                        hist_times
                    )
                    # print("current_time",current_time)
                    # print(true_next_codes)
                    # print(accumulated_codes[-5:])
                    repr = model.history_repr(event_representations, hist_times, current_time)

                    if repr is not None:
                        # 预测诊断代码并计算指标
                        code_probs = model.code_prediction_network(demographic, repr.unsqueeze(0))
                        code_probs = torch.sigmoid(code_probs)
                        visit_metrics = compute_prediction_metrics(
                            code_probs,
                            true_next_codes,
                            k_values
                        )

                        for metric_name, value in visit_metrics.items():
                            all_metrics[metric_name].append(value)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: OOM in evaluation, cleaning memory and continuing...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    # 计算代码预测指标的平均值
    avg_metrics = {
        metric_name: np.mean(values) for metric_name, values in all_metrics.items()
    }

    # 添加时间预测的评估指标
    if time_errors:
        avg_metrics['mean_time_error'] = np.mean(time_errors)
        avg_metrics['median_time_error'] = np.median(time_errors)
        avg_metrics['mean_relative_error'] = np.mean(relative_errors)
        avg_metrics['median_relative_error'] = np.median(relative_errors)

    # 获取最佳F1及其对应的k值
    f1_metrics = {k: avg_metrics[f'f1@{k}'] for k in k_values}
    best_k = max(f1_metrics.items(), key=lambda x: x[1])
    avg_metrics['best_f1'] = best_k[1]
    avg_metrics['best_f1_k'] = best_k[0]

    # 返回结果字典
    result = {
        'metrics': avg_metrics,  # 代码预测的平均指标
        'time_errors': time_errors,  # 原始的时间误差数据
        'relative_errors': relative_errors  # 原始的相对误差数据
    }

    return result

def should_save_model(current_metrics: Dict[str, float],
                      best_metrics: Dict[str, float],
                      k_values: List[int]) -> bool:
    """
    根据F1分数和时间预测误差来决定是否保存模型
    Args:
        current_metrics: 当前轮次的评估指标
        best_metrics: 历史最佳评估指标
        k_values: 使用的k值列表
    Returns:
        bool: 是否应该保存当前模型
    """
    # 计算当前轮次的综合得分
    current_f1_avg = np.mean([current_metrics[f'f1@{k}'] for k in k_values])
    current_time_error = current_metrics.get('median_relative_error', 0)
    current_score = current_f1_avg - 0.1 * current_time_error

    # 如果是第一个epoch（best_metrics为空）
    if not best_metrics:
        return True

    # 计算历史最佳的综合得分
    best_f1_avg = np.mean([best_metrics.get(f'f1@{k}', 0) for k in k_values])
    best_time_error = best_metrics.get('median_relative_error', 0)
    best_score = best_f1_avg - 0.1 * best_time_error

    return current_score > best_score