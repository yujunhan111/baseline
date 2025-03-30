import pickle
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import gc
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
from mappings import create_clean_mapping, create_all_mappings
from model import EHRModel
from filter_patients import filter_valid_patients
from get_patient_data import PatientDataset
from custom_collate import custom_collate
from tool import check_patients_data
from evaluation import evaluate_model,should_save_model
from collections import defaultdict


def validate_model(model, val_data_paths, mappings, index_set, batch_size, sample_points, k_values):
    """
    在验证集上评估模型性能
    """
    model.eval()
    # 合并所有验证数据
    print("\nLoading and merging validation data...")
    combined_val_data = {}
    for val_path in val_data_paths:
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
            filtered_val_data = filter_valid_patients(val_data)
            combined_val_data.update(filtered_val_data)
    #combined_val_data = dict(list(combined_val_data.items())[:500])
    print(f"\nTotal number of validation patients: {len(combined_val_data)}")

    # 创建合并后的数据加载器
    val_dataset = PatientDataset(combined_val_data, mappings, index_set, sample_points)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = tqdm(val_loader, desc="Validating")
    # 直接使用evaluate_model进行评估
    result = evaluate_model(model, val_loader, k_values)

    return result['metrics']


def train_epoch(model, train_loader, optimizer, scaler, batch_size, gradient_accumulation_steps, file_idx, total_files):
    """
    训练一个epoch
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器
        scaler: 梯度缩放器
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
    Returns:
        float: 平均损失值
    """
    model.train()
    batch_losses = []
    patient_count = 0
    running_loss = 0

    # 使用tqdm创建进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f'Training File [{file_idx}/{total_files}]', leave=True)

    for batch_idx, batch in pbar:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                #check_patients_data(batch)
                loss,code_loss,time_loss = model.compute_batch(batch)
                scaled_loss = loss / gradient_accumulation_steps
            # 反向传播
            scaler.scale(scaled_loss).backward()
            # 更新running loss
            running_loss += loss.item()
            patient_count += batch_size
            del batch
            # 梯度累积更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
            batch_losses.append(loss.item())
            # 每100个病人打印一次平均loss
            if patient_count >= 100:
                avg_loss = running_loss / patient_count
                #print(f"\nAverage loss over last {patient_count} patients: {avg_loss:.4f}")
                running_loss = 0
                patient_count = 0
            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'code_loss': f'{code_loss.item():.4f}',
                'time_loss': f'{time_loss.item():.4f}',
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
    pbar.close()
    return sum(batch_losses) / len(batch_losses) if batch_losses else 0


def train_model(train_data_paths, val_data_paths, code_dict, mappings, batch_size, num_epochs, learning_rate,
                gradient_accumulation_steps, sample_points, patience, k_values,lr_weight):
    """
    训练EHR模型的主函数
    """
    index_set = set(code_dict["index"])
    model = EHRModel(code_dict, demo_dim=70)
    model = model.cuda()
    model = torch.compile(model, mode="reduce-overhead")
    scaler = torch.cuda.amp.GradScaler()
    time_weight_params = []
    other_params = []
    # 遍历模型的所有参数进行分组
    for name, param in model.named_parameters():
        if 'time_weight_net' in name:
            time_weight_params.append(param)
        else:
            other_params.append(param)
    # 创建参数组
    param_groups = [
        {'params': other_params, 'lr': learning_rate},
        {'params': time_weight_params, 'lr': learning_rate * lr_weight}
    ]
    optimizer = optim.Adam(param_groups)
    # 基于综合得分的学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    best_metrics = defaultdict(float)
    best_score = float('-inf')
    patience_counter = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_train_loss = 0
            epoch_batches = 0

            # 训练阶段
            for file_idx, train_data_path in enumerate(train_data_paths, 1):
                with open(train_data_path, 'rb') as f:
                    train_data = pickle.load(f)
                    filtered_train_data = filter_valid_patients(train_data)
                    #filtered_train_data = dict(list(filtered_train_data.items())[:100])

                    dataset = PatientDataset(filtered_train_data, mappings, index_set, sample_points)
                    train_loader = DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=custom_collate
                    )
                    train_loss = train_epoch(
                        model,
                        train_loader,
                        optimizer,
                        scaler,
                        batch_size,
                        gradient_accumulation_steps,
                        file_idx,
                        len(train_data_paths)
                    )
                    epoch_train_loss += train_loss
                    epoch_batches += 1
                    # 在每个文件训练结束后打印time_weight_net的学习率
                    print(f"\nFile {file_idx} completed. Time weight net lr: {optimizer.param_groups[1]['lr']:.2e}")
            # 验证阶段
            print(f"\nValidation Phase (Epoch {epoch + 1}):")
            val_metrics = validate_model(
                model,
                val_data_paths,
                mappings,
                index_set,
                batch_size,
                sample_points,
                k_values
            )

            # 打印评估结果
            for k in k_values:
                print(f"k={k}: F1={val_metrics[f'f1@{k}']:.4f}, "
                      f"Precision={val_metrics[f'precision@{k}']:.4f}, "
                      f"Recall={val_metrics[f'recall@{k}']:.4f}")

            # 计算综合得分
            current_f1_avg = np.mean([val_metrics[f'f1@{k}'] for k in k_values])
            current_time_error = val_metrics.get('median_relative_error', 0)
            current_score = current_f1_avg - 0.2 *current_time_error

            # 打印详细指标
            print(
                f"Current score: {current_score:.4f} "
                f"(F1_avg: {current_f1_avg:.4f}, "
                f"median_relative_time_error: {current_time_error:.4f})"
            )
            if best_score != float('-inf'):
                print(f"Best score so far: {best_score:.4f}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            # 更新学习率
            scheduler.step(current_score)

            # 判断是否保存模型
            if should_save_model(val_metrics, best_metrics, k_values):
                best_metrics = val_metrics.copy()
                best_score = current_score
                patience_counter = 0

                # 保存模型和指标
                save_path = f'时间正则_best_model_score_{current_score:.4f}_epoch_{epoch + 1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'metrics': val_metrics,
                    'best_metrics': best_metrics,
                    'best_score': best_score,
                    'k_values': k_values,
                    'config': {
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'sample_points': sample_points
                    }
                }, save_path)
                print(f"Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping!")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        interrupt_save_path = f'interrupted_model_{current_time}.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': val_metrics if 'val_metrics' in locals() else None,
            'best_metrics': best_metrics,
            'best_score': best_score,
            'current_score': current_score if 'current_score' in locals() else 0,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'scaler': scaler.state_dict(),
            'timestamp': current_time,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'batch_size': batch_size,
            'sample_points': sample_points,
            'k_values': k_values
        }, interrupt_save_path)

        print(f"[{current_time}] Training interrupted at epoch {epoch + 1}")
        print(f"Saved to: {interrupt_save_path}")
        print(f"Current score: {current_score if 'current_score' in locals() else 'N/A'}")
        print(f"Best score: {best_score}")



def main(batch_size, num_epochs, learning_rate, gradient_accumulation_steps, sample_points, patience,lr_weight):
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 定义k值列表
    k_values = [5, 10, 15, 20, 25, 30,50,100]
    # 定义训练数据文件路径
    train_data_paths = [f"data/patient_records_batch_{i}.pkl" for i in range(1, 15)]
    # 加载验证数据
    val_data_paths = [
        "data/patient_records_batch_15.pkl",
        "data/patient_records_batch_16.pkl"
    ]
    # 加载人口统计数据
    directory_path = r"data/patients_dict.csv"
    patient = pd.read_csv(directory_path)
    mappings = create_all_mappings(patient)
    # 加载词嵌入数据
    directory_path = r"data/code-dict-biobert.parquet"
    code_dict = pd.read_parquet(directory_path)

    train_model(
        train_data_paths,
        val_data_paths,
        code_dict,
        mappings,
        batch_size,
        num_epochs,
        learning_rate,
        gradient_accumulation_steps,
        sample_points,
        patience,
        k_values,
        lr_weight
    )


if __name__ == "__main__":
    main(batch_size=1, num_epochs=200, learning_rate=0.0001, gradient_accumulation_steps=16, sample_points=5,
         patience=3,lr_weight =5)
