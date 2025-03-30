import pickle
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import gc
import json
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# 导入自定义模块
from LSTM import LSTMModelFull
from get_patient_data_diease import PatientDataset_full
from custom_collate_disease import (custom_collate_full)
from mappings import create_clean_mapping, create_all_mappings
from disease_codes import DISEASE_CODES, disease_weights
from death_weight import death_los_weights
from filter_patients import filter_valid_patients
from evaluation_disease import evaluate_model_full, evaluate_epoch_full, print_metrics_summary_full


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auroc = None
        self.early_stop = False
        self.best_model = None
        self.improvement_epochs = []  # 记录性能提升的epochs

    def __call__(self, val_auroc, model, epoch):
        if self.best_auroc is None:
            self.best_auroc = val_auroc
            self.improvement_epochs.append(epoch)
            self.save_checkpoint(model)
            return False

        if val_auroc > self.best_auroc + self.min_delta:
            self.best_auroc = val_auroc
            self.improvement_epochs.append(epoch)
            self.save_checkpoint(model)
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def save_checkpoint(self, model):
        """保存最佳模型状态"""
        import copy
        self.best_model = copy.deepcopy(model.state_dict())


def create_dataloader(dataset, batch_size=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_full  # 修改为新的collate函数
    )


def train_full_model(train_data_paths, val_data_paths, mappings, death_los_df, batch_size,
                     num_epochs, learning_rate, gradient_accumulation_steps, patience,
                     experiment_dir="experiments"):
    # 创建基础实验目录
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # 创建以当前时间命名的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_subdir = os.path.join(experiment_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_subdir)

    # 记录实验参数
    experiment_params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "patience": patience,
        "train_data_paths": train_data_paths,
        "val_data_paths": val_data_paths,
        "timestamp": timestamp
    }

    with open(os.path.join(experiment_subdir, "experiment_config.json"), "w") as f:
        json.dump(experiment_params, f, indent=4)

    # 创建模型保存目录
    models_dir = os.path.join(experiment_subdir, "models")
    os.makedirs(models_dir)

    # 创建结果目录
    results_dir = os.path.join(experiment_subdir, "results")
    os.makedirs(results_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载 code_dict
    code_dict_path = r"C:\Users\yujun\PycharmProjects\ehr_迁移\pic\data\code-dict-with-embedding-pic.parquet"
    code_dict = pd.read_parquet(code_dict_path)
    max_code_index = max(code_dict['index'])
    print(f"Maximum code index: {max_code_index}")

    # 提取 index_set
    if 'index' not in code_dict.columns:
        raise ValueError("code_dict 中缺少 'index' 列，请检查文件内容。")
    index_set = set(code_dict['index'].astype(int))  # 确保 index 是整数类型
    print(f"Loaded index_set with {len(index_set)} unique indices.")

    num_codes = 3000
    death_los_types = ['death', 'los_gt_7']

    # 初始化模型，使用扩展的模型类
    model = LSTMModelFull(
        num_codes=num_codes,
        hidden_dim=128,
        disease_names=list(DISEASE_CODES.keys()),
        demographic_dim=17,
        death_los_types=death_los_types
    ).to(device)

    # 保存模型结构信息
    with open(os.path.join(experiment_subdir, "model_structure.txt"), "w") as f:
        f.write(str(model))

    # 合并疾病权重和死亡/LOS权重
    combined_weights = {**disease_weights}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_avg_auroc = 0
    no_improvement = 0
    improvement_epochs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        total_batches = 0

        # 训练循环
        for file_idx, train_path in enumerate(train_data_paths, 1):
            try:
                print(f"\nProcessing file {file_idx}/{len(train_data_paths)}: {train_path}")

                # 加载和过滤数据
                with open(train_path, 'rb') as f:
                    current_data = pickle.load(f)
                filtered_data = filter_valid_patients(current_data)
                print(f"Filtered patients: {len(filtered_data)} / {len(current_data)}")

                # 创建训练数据集，传递死亡和LOS数据
                train_dataset = PatientDataset_full(filtered_data, mappings, index_set, death_los_df)
                train_loader = create_dataloader(train_dataset, batch_size)

                running_loss = 0.0
                n_batches = 0

                progress_bar = tqdm(total=len(filtered_data),
                                    desc=f"Processing patients in file {file_idx}",
                                    unit="patient",
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, Loss: {postfix[0]:.4f}]',
                                    postfix=[0.0])

                for batch_idx, batch in enumerate(train_loader):
                    try:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}

                        # 使用修改后的模型前向传播，传入疾病权重和死亡/LOS权重
                        batch_loss = model(batch,
                                           disease_weights=combined_weights,
                                           death_los_weights=death_los_weights,
                                           is_training=True)

                        if batch_loss > 0:  # 只在有有效样本时更新
                            batch_loss = batch_loss / gradient_accumulation_steps
                            batch_loss.backward()

                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                optimizer.zero_grad()

                            running_loss += batch_loss.item()
                            n_batches += 1
                            current_avg_loss = running_loss / n_batches
                            progress_bar.postfix[0] = current_avg_loss
                            epoch_loss += batch_loss.item()
                            total_batches += 1

                        progress_bar.update(batch_size)

                    except Exception as e:
                        print(f"Error in batch processing: {str(e)}")
                        continue

                progress_bar.close()
                del train_dataset, train_loader, filtered_data
                gc.collect()

            except Exception as e:
                print(f"Error processing file {train_path}: {str(e)}")
                continue

        # 验证阶段
        print("\nStarting validation...")
        model.eval()

        # 加载验证数据
        val_data = {}
        for val_path in val_data_paths:
            with open(val_path, 'rb') as f:
                val_data.update(pickle.load(f))
        filtered_val_data = filter_valid_patients(val_data)
        filtered_val_data = dict(list(filtered_val_data.items()))  # 使用所有验证数据
        print(f"Validation data size: {len(filtered_val_data)}")

        # 创建验证数据集，传递死亡和LOS数据
        val_dataset = PatientDataset_full(filtered_val_data, mappings, index_set, death_los_df)
        val_loader = create_dataloader(val_dataset, batch_size)

        # 使用新的评估函数
        val_metrics = evaluate_epoch_full(model, val_loader, device)

        # 将评估结果写入文件
        with open(os.path.join(results_dir, f"metrics_epoch_{epoch + 1}.json"), "w") as f:
            # 转换metrics中的numpy类型为Python原生类型
            serializable_metrics = {
                "disease_metrics": {
                    disease: {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else (
                            int(v) if isinstance(v, (np.int32, np.int64)) else v
                        ) for k, v in metrics.items()
                    } for disease, metrics in val_metrics["disease_metrics"].items()
                },
                "death_los_metrics": {
                    task: {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else (
                            int(v) if isinstance(v, (np.int32, np.int64)) else v
                        ) for k, v in metrics.items()
                    } for task, metrics in val_metrics["death_los_metrics"].items()
                },
                "disease_average_metrics": {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else (
                        int(v) if isinstance(v, (np.int32, np.int64)) else v
                    ) for k, v in val_metrics["disease_average_metrics"].items()
                }
            }
            json.dump(serializable_metrics, f, indent=4)

        # 打印详细的验证结果
        print_metrics_summary_full(val_metrics)

        # 计算平均AUROC（考虑疾病和死亡/LOS预测）
        disease_aurocs = [
            metrics['auroc']
            for disease, metrics in val_metrics['disease_metrics'].items()
            if disease in DISEASE_CODES and not np.isnan(metrics['auroc'])
        ]

        # 加上死亡和LOS的AUROC
        death_los_aurocs = [
            metrics['auroc']
            for task, metrics in val_metrics['death_los_metrics'].items()
            if not np.isnan(metrics['auroc'])
        ]

        # 计算总平均AUROC
        all_aurocs = disease_aurocs + death_los_aurocs
        current_avg_auroc = np.mean(all_aurocs) if all_aurocs else 0

        print(f"\nEpoch {epoch + 1}")
        print(f"Current Average AUROC: {current_avg_auroc:.4f}")
        print(f"Best Average AUROC: {best_avg_auroc:.4f}")

        # 检查是否有性能提升
        if current_avg_auroc > best_avg_auroc:
            best_avg_auroc = current_avg_auroc
            improvement_epochs.append(epoch + 1)
            no_improvement = 0

            # 只在性能提升时保存模型
            save_path = os.path.join(models_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_auroc': current_avg_auroc,
                'val_metrics': serializable_metrics
            }, save_path)
            print(f"Model saved at epoch {epoch + 1} with improved AUROC: {current_avg_auroc:.4f}")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs")

        # Early stopping 检查
        if no_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best AUROC achieved: {best_avg_auroc:.4f}")
            print(f"Performance improved at epochs: {improvement_epochs}")
            break

        # 清理验证数据
        del val_dataset, val_loader, filtered_val_data, val_data
        gc.collect()

    # 训练结束，记录最终结果
    final_results = {
        "best_avg_auroc": best_avg_auroc,
        "improvement_epochs": improvement_epochs,
        "total_epochs_trained": epoch + 1,
        "early_stopped": no_improvement >= patience
    }

    with open(os.path.join(experiment_subdir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    return model, best_avg_auroc, improvement_epochs, experiment_subdir


def main(batch_size, num_epochs, learning_rate, gradient_accumulation_steps, patience):
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 数据路径
    train_data_paths = [rf"C:\Users\yujun\Desktop\patient_records_pic\patient_records_batch_{i}.pkl" for i in
                        range(1, 8)]
    val_data_paths = [
        r"C:\Users\yujun\Desktop\patient_records_pic\patient_records_batch_8.pkl",
        r"C:\Users\yujun\Desktop\patient_records_pic\patient_records_batch_9.pkl",
        r"C:\Users\yujun\Desktop\patient_records_pic\patient_records_batch_10.pkl",
    ]

    # 加载人口统计学数据
    directory_path = r"C:\Users\yujun\PycharmProjects\ehr_迁移\pic\数据预处理\records\patients_dict.csv"
    patient = pd.read_csv(directory_path)
    mappings = create_all_mappings(patient)

    # 加载死亡和LOS数据
    death_los_path = r"C:\Users\yujun\Desktop\patient_records_pic\death_los_pic.csv"
    death_los_df = pd.read_csv(death_los_path)
    print(f"Loaded death and LOS data with {len(death_los_df)} records")

    # 创建实验目录
    experiment_dir = "experiments"
    os.makedirs(experiment_dir, exist_ok=True)

    # 训练模型并获取实验目录
    model, best_auroc, improvement_epochs, experiment_path = train_full_model(
        train_data_paths,
        val_data_paths,
        mappings,
        death_los_df,
        batch_size,
        num_epochs,
        learning_rate,
        gradient_accumulation_steps,
        patience,
        experiment_dir
    )

    print(f"实验完成！结果保存在: {experiment_path}")


if __name__ == "__main__":
    main(batch_size=1, num_epochs=200, learning_rate=0.0001, gradient_accumulation_steps=16, patience=3)