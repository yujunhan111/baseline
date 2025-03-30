from disease_codes import DISEASE_CODES
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from tool import get_current_demographic


def evaluate_model_full(model, val_loader, device):
    """评估模型在疾病、死亡和LOS预测任务上的性能"""
    model.eval()

    # 初始化存储预测结果的字典
    disease_metrics = {
        disease_name: {'y_true': [], 'y_score': []}
        for disease_name in DISEASE_CODES.keys()
    }

    death_los_metrics = {
        'death': {'y_true': [], 'y_score': []},
        'los_gt_7': {'y_true': [], 'y_score': []}
    }

    with torch.no_grad():
        for batch in val_loader:
            demographic_features = batch['demographic'].to(device)
            disease_data = batch['disease_data']
            death_los_data = batch['death_los_data']

            for patient_idx in range(len(demographic_features)):
                demo_feat = demographic_features[patient_idx].unsqueeze(0)

                # 处理疾病预测
                for disease_name, disease_info in disease_data[patient_idx].items():
                    if disease_info['label'] == -1 or not disease_info['history_codes']:
                        continue

                    hist_embeddings, hist_times_tensor = model.get_code_embeddings(
                        disease_info['history_codes'], disease_info['history_times']
                    )

                    event_representations = model.pretrained_model.attention(
                        hist_embeddings, hist_embeddings, hist_times_tensor
                    )

                    repr = model.pretrained_model.history_repr(
                        event_representations, hist_times_tensor, disease_info['event_time']
                    )

                    if repr is not None:
                        demographic_current = get_current_demographic(demo_feat, disease_info['event_time'])
                        pred = model.disease_classifiers[disease_name](
                            demographic_current, repr.unsqueeze(0)
                        )

                        score = torch.sigmoid(pred).item()
                        true_label = 1 if disease_info['label'] > 0.5 else 0

                        disease_metrics[disease_name]['y_true'].append(true_label)
                        disease_metrics[disease_name]['y_score'].append(score)

                # 处理死亡预测
                if 'death' in death_los_data[patient_idx]:
                    death_info = death_los_data[patient_idx]['death']
                    if death_info['label'] != -1 and death_info['history_codes']:
                        hist_embeddings, hist_times_tensor = model.get_code_embeddings(
                            death_info['history_codes'], death_info['history_times']
                        )

                        event_representations = model.pretrained_model.attention(
                            hist_embeddings, hist_embeddings, hist_times_tensor
                        )

                        repr = model.pretrained_model.history_repr(
                            event_representations, hist_times_tensor, death_info['event_time']
                        )

                        if repr is not None:
                            demographic_current = get_current_demographic(demo_feat, death_info['event_time'])
                            pred = model.death_classifier(
                                demographic_current, repr.unsqueeze(0)
                            )

                            score = torch.sigmoid(pred).item()
                            true_label = 1 if death_info['label'] > 0.5 else 0

                            death_los_metrics['death']['y_true'].append(true_label)
                            death_los_metrics['death']['y_score'].append(score)

                # 处理LOS预测
                if 'los_gt_7' in death_los_data[patient_idx]:
                    los_info = death_los_data[patient_idx]['los_gt_7']
                    if los_info['label'] != -1 and los_info['history_codes']:
                        hist_embeddings, hist_times_tensor = model.get_code_embeddings(
                            los_info['history_codes'], los_info['history_times']
                        )

                        event_representations = model.pretrained_model.attention(
                            hist_embeddings, hist_embeddings, hist_times_tensor
                        )

                        repr = model.pretrained_model.history_repr(
                            event_representations, hist_times_tensor, los_info['event_time']
                        )

                        if repr is not None:
                            demographic_current = get_current_demographic(demo_feat, los_info['event_time'])
                            pred = model.los_classifier(
                                demographic_current, repr.unsqueeze(0)
                            )

                            score = torch.sigmoid(pred).item()
                            true_label = 1 if los_info['label'] > 0.5 else 0

                            death_los_metrics['los_gt_7']['y_true'].append(true_label)
                            death_los_metrics['los_gt_7']['y_score'].append(score)

    return disease_metrics, death_los_metrics


def calculate_metrics_full(disease_metrics, death_los_metrics):
    """计算各任务的AUROC、PR-AUC和F1分数"""
    results = {
        'disease_metrics': {},
        'death_los_metrics': {},
        'overall_score': 0.0
    }

    valid_aurocs = []

    # 处理疾病指标
    for disease_name, metrics in disease_metrics.items():
        if len(metrics['y_true']) > 0 and len(np.unique(metrics['y_true'])) > 1:
            try:
                auroc = roc_auc_score(metrics['y_true'], metrics['y_score'])
                precision, recall, _ = precision_recall_curve(metrics['y_true'], metrics['y_score'])
                pr_auc = auc(recall, precision)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1 = np.max(f1_scores)

                results['disease_metrics'][disease_name] = {
                    'auroc': auroc,
                    'pr_auc': pr_auc,
                    'f1': best_f1,
                    'samples': len(metrics['y_true']),
                    'positives': sum(metrics['y_true'])
                }

                valid_aurocs.append(auroc)
            except:
                results['disease_metrics'][disease_name] = {
                    'auroc': float('nan'),
                    'pr_auc': float('nan'),
                    'f1': float('nan'),
                    'samples': len(metrics['y_true']),
                    'positives': sum(metrics['y_true'])
                }

    # 处理死亡和LOS指标
    for task, metrics in death_los_metrics.items():
        if len(metrics['y_true']) > 0 and len(np.unique(metrics['y_true'])) > 1:
            try:
                auroc = roc_auc_score(metrics['y_true'], metrics['y_score'])
                precision, recall, _ = precision_recall_curve(metrics['y_true'], metrics['y_score'])
                pr_auc = auc(recall, precision)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1 = np.max(f1_scores)

                results['death_los_metrics'][task] = {
                    'auroc': auroc,
                    'pr_auc': pr_auc,
                    'f1': best_f1,
                    'samples': len(metrics['y_true']),
                    'positives': sum(metrics['y_true'])
                }

                valid_aurocs.append(auroc)
            except:
                results['death_los_metrics'][task] = {
                    'auroc': float('nan'),
                    'pr_auc': float('nan'),
                    'f1': float('nan'),
                    'samples': len(metrics['y_true']),
                    'positives': sum(metrics['y_true'])
                }

    # 计算总体得分
    results['overall_score'] = np.mean(valid_aurocs) if valid_aurocs else float('nan')
    results['valid_tasks'] = len(valid_aurocs)
    results['total_tasks'] = len(disease_metrics) + len(death_los_metrics)

    return results


def print_metrics_summary_full(metrics):
    """打印评估指标摘要"""
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Overall AUROC: {metrics['overall_score']:.4f}")
    print(f"Valid tasks: {metrics['valid_tasks']}/{metrics['total_tasks']}")

    print("\n--- Disease Prediction ---")
    for disease_name, disease_metrics in metrics['disease_metrics'].items():
        auroc = disease_metrics['auroc']
        if not np.isnan(auroc):
            print(f"{disease_name}: AUROC={auroc:.4f}, F1={disease_metrics['f1']:.4f}, "
                  f"Samples={disease_metrics['samples']} (Pos: {disease_metrics['positives']})")

    print("\n--- Death & LOS Prediction ---")
    for task, task_metrics in metrics['death_los_metrics'].items():
        auroc = task_metrics['auroc']
        if not np.isnan(auroc):
            print(f"{task}: AUROC={auroc:.4f}, F1={task_metrics['f1']:.4f}, "
                  f"Samples={task_metrics['samples']} (Pos: {task_metrics['positives']})")


def evaluate_epoch_full(model, val_loader, device):
    """单个epoch的评估函数 (用于train_disease.py)"""
    disease_metrics, death_los_metrics = evaluate_model_full(model, val_loader, device)
    return calculate_metrics_full(disease_metrics, death_los_metrics)