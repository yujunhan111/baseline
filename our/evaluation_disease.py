from disease_codes import DISEASE_CODES
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from death_weight import death_weights
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model_disease(model, val_loader):
    """
    评估疾病预测模型的性能

    Args:
        model: 训练好的模型
        val_loader: 验证数据加载器

    Returns:
        dict: 包含每种疾病的评估指标
    """
    model.eval()
    # 初始化用于存储每种疾病预测结果的字典
    disease_metrics = {
        disease_name: {
            'y_true': [],
            'y_score': []
        } for disease_name in DISEASE_CODES.keys()
    }
    # 初始化用于存储死亡预测结果的字典
    death_metrics = {
        death_type: {
            'y_true': [],
            'y_score': []
        } for death_type in death_weights.keys()
    }

    with torch.no_grad():
        for batch in val_loader:
            demographic_features = batch['demographic'].cuda()
            disease_data = batch['disease_data']
            death_labels = batch['death_labels']
            # 对批次中的每个患者进行预测
            for patient_idx in range(len(demographic_features)):
                demo_feat = demographic_features[patient_idx].unsqueeze(0)
                patient_diseases = disease_data[patient_idx]
                patient_death = death_labels[patient_idx]
                # 对每种疾病进行预测
                for disease_name, disease_info in patient_diseases.items():
                    label = disease_info['label']

                    # 跳过未定义的标签
                    if label == -1:
                        continue

                    # 获取历史数据
                    hist_codes = disease_info['history_codes']
                    hist_times = disease_info['history_times']
                    event_time = disease_info['event_time']

                    # 跳过没有历史记录的情况
                    if not hist_codes or not hist_times:
                        continue

                    # 获取编码嵌入
                    hist_embeddings, hist_times_tensor = model.get_code_embeddings(
                        hist_codes, hist_times
                    )

                    # 获取事件表示
                    event_representations = model.pretrained_model.attention(
                        hist_embeddings,
                        hist_embeddings,
                        hist_times_tensor
                    )

                    # 获取最终表示
                    repr = model.pretrained_model.history_repr(
                        event_representations,
                        hist_times_tensor,
                        event_time
                    )

                    if repr is not None:
                        # 对该疾病进行预测
                        pred = model.disease_classifiers[disease_name](
                            demo_feat,
                            repr.unsqueeze(0)
                        )

                        # 获取预测分数
                        score = torch.sigmoid(pred).item()
                        true_label = 1 if label > 0.5 else 0

                        # 存储预测结果
                        disease_metrics[disease_name]['y_true'].append(true_label)
                        disease_metrics[disease_name]['y_score'].append(score)

                # 处理死亡预测评估
                for death_type, death_info in patient_death.items():
                    label = death_info['label']
                    hist_codes = death_info['history_codes']
                    hist_times = death_info['history_times']
                    event_time = death_info['event_time']

                    # 跳过没有历史记录的情况
                    if not hist_codes or not hist_times:
                        continue
                    if label == -1:
                        continue

                    # 获取编码嵌入
                    hist_embeddings, hist_times_tensor = model.get_code_embeddings(
                        hist_codes, hist_times
                    )

                    # 获取事件表示
                    event_representations = model.pretrained_model.attention(
                        hist_embeddings,
                        hist_embeddings,
                        hist_times_tensor
                    )
                    # 获取最终表示
                    repr = model.pretrained_model.history_repr(
                        event_representations,
                        hist_times_tensor,
                        event_time
                    )
                    if repr is not None:
                        # 对该死亡类型进行预测
                        pred = model.death_classifiers[death_type](
                            demo_feat,
                            repr.unsqueeze(0)
                        )
                        # 获取预测分数
                        score = torch.sigmoid(pred).item()
                        true_label = 1 if label > 0.5 else 0
                        # 存储预测结果
                        death_metrics[death_type]['y_true'].append(true_label)
                        death_metrics[death_type]['y_score'].append(score)
    return disease_metrics, death_metrics


def calculate_disease_metrics(disease_metrics, death_metrics):
    """
    计算每种疾病和死亡类型的AUC-ROC和总分

    Args:
        disease_metrics (dict): evaluate_model返回的疾病指标字典
        death_metrics (dict): evaluate_model返回的死亡指标字典

    Returns:
        dict: 包含疾病和死亡的AUC-ROC和总分
    """
    # 初始化结果字典
    final_metrics = {
        'disease': {},
        'death': {}
    }

    # 初始化计数器
    disease_total_score = 0
    death_total_score = 0
    valid_disease_count = 0
    valid_death_count = 0

    # 处理疾病指标
    for disease_name, metrics in disease_metrics.items():
        y_true = metrics['y_true']
        y_score = metrics['y_score']

        sample_count = len(y_true)
        positive_count = sum(y_true)

        # 检查是否只有一个类别
        if len(np.unique(y_true)) > 1:
            try:
                # 计算ROC-AUC
                roc_auc = roc_auc_score(y_true, y_score)
                precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1 = np.max(f1_scores)
                best_index = np.argmax(f1_scores)
                best_precision = precision[best_index]
                best_recall = recall[best_index]
                recall = np.max(recall)
                precision = np.max(precision)

                # y_pred = (y_score > 0.5).astype(int)
                # acc = accuracy_score(y_true, y_pred)
                # prec = precision_score(y_true, y_pred, zero_division=0)
                # rec = recall_score(y_true, y_pred)
                # f1 = f1_score(y_true, y_pred)

                final_metrics['disease'][disease_name] = {
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    # 'accuracy': acc,
                    'best_f1': best_f1,
                    'best_precision': best_precision,
                    'best_recall': best_recall,
                    'sample_count': len(y_true),
                    'positive_count': sum(y_true)
                }

            except Exception as e:
                print(f"Error calculating metrics for disease {disease_name}: {str(e)}")
                final_metrics['disease'][disease_name] = {
                    'roc_auc': float('nan'),
                    'pr_auc': float('nan'),
                    'best_f1': float('nan'),
                    'best_precision': float('nan'),
                    'best_recall': float('nan'),
                    'sample_count': sample_count,
                    'positive_count': positive_count
                }
        else:
            print(f"Warning: disease {disease_name} has only one class in predictions")
            final_metrics['disease'][disease_name] = {
                'roc_auc': float('nan'),
                'pr_auc': float('nan'),
                'best_f1': float('nan'),
                'best_precision': float('nan'),
                'best_recall': float('nan'),
                'sample_count': sample_count,
                'positive_count': positive_count
            }

    # 处理死亡指标
    for death_type, metrics in death_metrics.items():
        y_true = metrics['y_true']
        y_score = metrics['y_score']

        sample_count = len(y_true)
        positive_count = sum(y_true)

        # 检查是否只有一个类别
        if len(np.unique(y_true)) > 1:
            try:
                # 计算ROC-AUC
                roc_auc = roc_auc_score(y_true, y_score)
                precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1 = np.max(f1_scores)
                best_index = np.argmax(f1_scores)
                best_precision = precision[best_index]
                best_recall = recall[best_index]
                recall = np.max(recall)
                precision = np.max(precision)

                # 累加有效的ROC-AUC分数
                death_total_score += roc_auc
                valid_death_count += 1

                final_metrics['death'][death_type] = {
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'best_f1': best_f1,
                    'best_precision': best_precision,
                    'best_recall': best_recall,
                    'sample_count': len(y_true),
                    'positive_count': sum(y_true)
                }
            except Exception as e:
                print(f"Error calculating metrics for death type {death_type}: {str(e)}")
                final_metrics['death'][death_type] = {
                    'roc_auc': float('nan'),
                    'pr_auc': float('nan'),
                    'best_f1': float('nan'),
                    'best_precision': float('nan'),
                    'best_recall': float('nan'),
                    'sample_count': sample_count,
                    'positive_count': positive_count
                }
        else:
            print(f"Warning: death type {death_type} has only one class in predictions")
            final_metrics['death'][death_type] = {
                'roc_auc': float('nan'),
                'pr_auc': float('nan'),
                'best_f1': float('nan'),
                'best_precision': float('nan'),
                'best_recall': float('nan'),
                'sample_count': sample_count,
                'positive_count': positive_count
            }

    # 返回结果
    return {
        'disease_metrics': final_metrics['disease'],
        'death_metrics': final_metrics['death'],
        'disease_total_roc_auc': disease_total_score,
        'death_total_roc_auc': death_total_score,
        'valid_disease_count': valid_disease_count,
        'valid_death_count': valid_death_count,
        'total_disease_count': len(disease_metrics),
        'total_death_count': len(death_metrics)
    }


def print_metrics_summary(metrics):
    """
    打印评估指标的摘要

    Args:
        metrics (dict): calculate_metrics返回的指标字典
    """
    print("\n=== Disease Prediction Metrics ===")
    print(f"Total Disease ROC-AUC: {metrics['disease_total_roc_auc']:.4f}")
    print(f"Valid Diseases: {metrics['valid_disease_count']}/{metrics['total_disease_count']}")

    print("\nDetails for Each Disease:")
    for disease_name, disease_metrics in metrics['disease_metrics'].items():
        print(f"\n{disease_name}:")
        print(f"  ROC-AUC: {disease_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {disease_metrics['pr_auc']:.4f}")
        print(f"  F1: {disease_metrics['best_f1']:.4f}")
        print(f"  Precision: {disease_metrics['best_precision']:.4f}")
        print(f"  Recall: {disease_metrics['best_recall']:.4f}")
        print(f"  Samples: {disease_metrics['sample_count']}")
        print(f"  Positive samples: {disease_metrics['positive_count']}")

    print("\n=== Death Prediction Metrics ===")
    print(f"Total Death ROC-AUC: {metrics['death_total_roc_auc']:.4f}")
    print(f"Valid Death Types: {metrics['valid_death_count']}/{metrics['total_death_count']}")

    print("\nDetails for Each Death Type:")
    for death_type, death_metrics in metrics['death_metrics'].items():
        print(f"\n{death_type}:")
        print(f"  ROC-AUC: {death_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {death_metrics['pr_auc']:.4f}")
        print(f"  F1: {death_metrics['best_f1']:.4f}")
        print(f"  Precision: {death_metrics['best_precision']:.4f}")
        print(f"  Recall: {death_metrics['best_recall']:.4f}")
        print(f"  Samples: {death_metrics['sample_count']}")
        print(f"  Positive samples: {death_metrics['positive_count']}")