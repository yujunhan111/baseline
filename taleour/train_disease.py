import pickle
import warnings;

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import gc
import json
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from get_patient_data_diease import PatientDataset_full
from custom_collate_disease import custom_collate_full
from mappings import create_clean_mapping, create_all_mappings
from model_disease import DiseaseModel
from filter_patients import filter_valid_patients
from evaluation_disease import evaluate_model_full, calculate_metrics_full, print_metrics_summary_full
from disease_codes import DISEASE_CODES
from death_weight import death_los_weights


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auroc = None
        self.early_stop = False
        self.best_model = None
        self.improvement_epochs = []

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


def train_disease_model(train_data_paths, val_data_paths, code_dict, mappings, batch_size, num_epochs, learning_rate,
                        gradient_accumulation_steps, patience, pretrained_model_path, death_los_df,
                        experiment_dir="experiments"):
    """迁移学习训练流程"""

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
        "pretrained_model_path": pretrained_model_path,
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

    # 提取index_set
    index_set = set(code_dict["index"])
    print(f"Loaded index_set with {len(index_set)} unique indices.")

    # 初始化模型
    model = DiseaseModel(
        pretrained_path=pretrained_model_path,
        code_dict=code_dict,
        use_adapter=True,
        input_dim=17
    ).to(device)

    # 保存模型结构信息
    with open(os.path.join(experiment_subdir, "model_structure.txt"), "w", encoding="utf-8") as f:
        f.write(str(model))

    # 选择性冻结参数
    model.partial_unfreeze()

    # 设置优化器
    pretrained_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'time_weight_net' in name or 'Q_base' in name:
                pretrained_params.append(param)
            elif 'adapter' in name:
                pretrained_params.append(param)
            else:
                other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': learning_rate},
        {'params': pretrained_params, 'lr': learning_rate * 0.1}
    ]

    optimizer = optim.Adam(param_groups)
    print("\nLearning rates:")
    print(f"Other params (e.g. classifiers): {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Pretrained params (time_weight_net & Q_base, adapter): {optimizer.param_groups[1]['lr']:.2e}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    early_stopping = EarlyStopping(patience=patience)

    try:
        for epoch in range(num_epochs):
            model.train()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
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

                    # 创建训练数据集
                    train_dataset = PatientDataset_full(filtered_data, mappings, index_set, death_los_df)
                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=custom_collate_full
                    )

                    running_loss = 0.0
                    n_batches = 0

                    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                        desc=f"Training File [{file_idx}/{len(train_data_paths)}]",
                                        leave=True,
                                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')

                    for batch_idx, batch in progress_bar:
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # 将数据移动到设备
                            batch = {
                                'demographic': batch['demographic'].to(device),
                                'disease_data': batch['disease_data'],
                                'death_los_data': batch['death_los_data']  # 新字段
                            }

                            # 计算损失
                            with torch.cuda.amp.autocast():
                                loss = model.compute_batch_loss(batch)
                                scaled_loss = loss / gradient_accumulation_steps

                            # 反向传播
                            scaled_loss.backward()

                            del batch

                            # 梯度累积更新
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                                torch.cuda.empty_cache()
                                gc.collect()

                            running_loss += loss.item()
                            n_batches += 1
                            epoch_loss += loss.item()
                            total_batches += 1

                            # 更新进度条
                            progress_bar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'avg_loss': f'{running_loss / n_batches:.4f}',
                                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                            })

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print(f"\nOOM at batch {batch_idx}, skipping...")
                                gc.collect()
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e

                    # 清理内存
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
            print(f"Validation data size: {len(filtered_val_data)}")

            # 创建验证数据集
            val_dataset = PatientDataset_full(filtered_val_data, mappings, index_set, death_los_df)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=custom_collate_full
            )

            # 评估模型
            val_loader = tqdm(val_loader, desc="Validating")
            disease_metrics, death_los_metrics = evaluate_model_full(model, val_loader, device)
            metrics = calculate_metrics_full(disease_metrics, death_los_metrics)
            print_metrics_summary_full(metrics)

            # 将评估结果写入文件
            with open(os.path.join(results_dir, f"metrics_epoch_{epoch + 1}.json"), "w") as f:
                # 转换metrics中的numpy类型为Python原生类型
                serializable_metrics = {
                    k: (float(v) if isinstance(v, (np.float32, np.float64)) else
                        (int(v) if isinstance(v, (np.int32, np.int64)) else v))
                    for k, v in metrics.items()
                }

                for category in ['disease_metrics', 'death_los_metrics']:
                    if category in serializable_metrics:
                        serializable_metrics[category] = {
                            k: {
                                kk: (float(vv) if isinstance(vv, (np.float32, np.float64)) else
                                     (int(vv) if isinstance(vv, (np.int32, np.int64)) else vv))
                                for kk, vv in v.items()
                            }
                            for k, v in serializable_metrics[category].items()
                        }

                json.dump(serializable_metrics, f, indent=4)

            # 计算平均AUROC
            disease_aurocs = [
                metrics['disease_metrics'][disease]['auroc']
                for disease in DISEASE_CODES.keys()
                if disease in metrics['disease_metrics'] and not np.isnan(metrics['disease_metrics'][disease]['auroc'])
            ]

            death_los_aurocs = [
                metrics['death_los_metrics'][task]['auroc']
                for task in ['death', 'los_gt_7']
                if task in metrics['death_los_metrics'] and not np.isnan(metrics['death_los_metrics'][task]['auroc'])
            ]

            all_aurocs = disease_aurocs + death_los_aurocs
            current_avg_auroc = np.mean(all_aurocs) if all_aurocs else 0

            print(f"\nEpoch {epoch + 1}")
            print(f"Current Average AUROC: {current_avg_auroc:.4f}")
            print(f"Best AUROC so far: {early_stopping.best_auroc if early_stopping.best_auroc is not None else 0:.4f}")

            # 更新学习率调度器
            scheduler.step(current_avg_auroc)

            # 检查早停
            if early_stopping(current_avg_auroc, model, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best AUROC achieved: {early_stopping.best_auroc:.4f}")
                print(f"Performance improved at epochs: {early_stopping.improvement_epochs}")

                # 保存最佳模型
                if early_stopping.best_model is not None:
                    best_model_path = os.path.join(models_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': early_stopping.best_model,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'avg_auroc': early_stopping.best_auroc,
                        'improvement_epochs': early_stopping.improvement_epochs
                    }, best_model_path)
                    print(f"Best model saved to {best_model_path}")

                break

            # 在性能提升时保存模型
            if epoch + 1 in early_stopping.improvement_epochs:
                save_path = os.path.join(models_dir, f'model_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'avg_auroc': current_avg_auroc
                }, save_path)
                print(f"Model saved at epoch {epoch + 1} with improved AUROC: {current_avg_auroc:.4f}")

            # 清理内存
            del val_dataset, val_loader, filtered_val_data, val_data
            gc.collect()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        interrupt_save_path = os.path.join(models_dir, f'interrupted_model_{current_time}.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics if 'metrics' in locals() else None,
            'best_auroc': early_stopping.best_auroc if hasattr(early_stopping, 'best_auroc') else None,
            'current_auroc': current_avg_auroc if 'current_avg_auroc' in locals() else 0,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'timestamp': current_time,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'batch_size': batch_size
        }, interrupt_save_path)

        print(f"[{current_time}] Training interrupted at epoch {epoch + 1}")
        print(f"Saved to: {interrupt_save_path}")
        print(f"Current AUROC: {current_avg_auroc if 'current_avg_auroc' in locals() else 'N/A'}")
        print(f"Best AUROC: {early_stopping.best_auroc if hasattr(early_stopping, 'best_auroc') else 'N/A'}")

    # 训练结束，记录最终结果
    final_results = {
        "best_avg_auroc": early_stopping.best_auroc if hasattr(early_stopping, 'best_auroc') else None,
        "improvement_epochs": early_stopping.improvement_epochs,
        "total_epochs_trained": epoch + 1,
        "early_stopped": early_stopping.early_stop
    }

    with open(os.path.join(experiment_subdir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    return model, early_stopping.best_auroc, early_stopping.improvement_epochs, experiment_subdir


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
    directory_path =  r"C:\Users\yujun\PycharmProjects\ehr_迁移\pic\数据预处理\records\patients_dict.csv"
    patient = pd.read_csv(directory_path)
    mappings = create_all_mappings(patient)

    # 加载词嵌入数据
    code_dict_path = r"C:\Users\yujun\PycharmProjects\ehr_迁移\pic\data\code-dict-with-embedding-pic.parquet"
    code_dict = pd.read_parquet(code_dict_path)

    # 加载死亡和LOS数据
    death_los_path = r"C:\Users\yujun\Desktop\patient_records_pic\death_los_pic.csv"
    death_los_df = pd.read_csv(death_los_path)
    print(f"Loaded death and LOS data with {len(death_los_df)} records")

    # 预训练模型路径
    pretrained_model_path = "时间正则_best_model_score_0.1412_epoch_1_20250127_234527.pt"

    # 创建实验目录
    experiment_dir = "experiments"
    os.makedirs(experiment_dir, exist_ok=True)

    # 训练模型
    model, best_auroc, improvement_epochs, experiment_path = train_disease_model(
        train_data_paths,
        val_data_paths,
        code_dict,
        mappings,
        batch_size,
        num_epochs,
        learning_rate,
        gradient_accumulation_steps,
        patience,
        pretrained_model_path,
        death_los_df,
        experiment_dir
    )

    print(f"实验完成！结果保存在: {experiment_path}")


if __name__ == "__main__":
    main(batch_size=1, num_epochs=200, learning_rate=0.0001, gradient_accumulation_steps=16, patience=3)