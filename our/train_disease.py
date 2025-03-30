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

from get_patient_data_diease import PatientDataset_disease
from custom_collate_disease import custom_collate_disease
from tool import check_patients_data_diease
from mappings import create_clean_mapping, create_all_mappings
from model_disease import DiseaseModel
from filter_patients import filter_valid_patients
from evaluation_disease import evaluate_model_disease,calculate_disease_metrics,print_metrics_summary
def train_disease_model(train_data_paths, val_data_paths, code_dict, mappings, batch_size, num_epochs, learning_rate,
                gradient_accumulation_steps, patience,pretrained_model_patch,death_df):
    """分两阶段训练"""
    best_score = float('-inf')
    index_set = set(code_dict["index"])
    model = DiseaseModel(pretrained_model_patch,code_dict)
    #选择冻结模型
    #model.freeze_pretrained()
    #选择解冻模型
    model.partial_unfreeze()
    model = model.cuda()
    model = torch.compile(model, mode="max-autotune")
    scaler = torch.cuda.amp.GradScaler()
    pretrained_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'time_weight_net' in name or 'Q_base' in name:
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
    print(f"Pretrained params (time_weight_net & Q_base): {optimizer.param_groups[1]['lr']:.2e}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    try:
        for epoch in range(num_epochs):
            model.train()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            for file_idx, train_data_path in enumerate(train_data_paths, 1):
                with open(train_data_path, 'rb') as f:
                    train_data = pickle.load(f)
                    filtered_train_data = filter_valid_patients(train_data)
                    #filtered_train_data = dict(list(filtered_train_data.items())[:3])
                    train_dataset = PatientDataset_disease(filtered_train_data, mappings, index_set,death_df)
                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=custom_collate_disease
                    )
                    train_loss = train_epoch(model, train_loader, optimizer, scaler, gradient_accumulation_steps, file_idx,
                                             len(train_data_paths))
            # 验证阶段
            print(f"\nValidation Phase (Epoch {epoch + 1}):")
            val_metrics = validate_model(
                model,
                val_data_paths,
                mappings,
                index_set,
                batch_size,
                death_df)
            current_score = val_metrics['disease_total_roc_auc']+val_metrics['death_total_roc_auc']
            scheduler.step(current_score)
            if current_score > best_score:
                best_score = current_score
                epochs_without_improvement = 0
                # 生成保存路径
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'微调_best_model_score_{current_score:.4f}_epoch_{epoch}_{current_time}.pt'

                # 保存完整的checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'score': current_score,
                    'best_score': best_score,
                    'metrics': val_metrics,
                    'training_params': {
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'batch_size': batch_size
                    },
                    'scaler': scaler.state_dict(),
                    'timestamp': current_time
                }

                # 保存模型
                torch.save(checkpoint, save_path)
            else:
                epochs_without_improvement += 1

            # Early stopping 检查
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best score achieved: {best_score:.4f}")
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
            'best_score': best_score,
            'current_score': current_score if 'current_score' in locals() else 0,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'scaler': scaler.state_dict(),
            'timestamp': current_time,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'batch_size': batch_size
        }, interrupt_save_path)

        print(f"[{current_time}] Training interrupted at epoch {epoch + 1}")
        print(f"Saved to: {interrupt_save_path}")
        print(f"Current score: {current_score if 'current_score' in locals() else 'N/A'}")
        print(f"Best score: {best_score}")

def train_epoch(model, train_loader, optimizer, scaler, gradient_accumulation_steps, file_idx, total_files):
    model.train()
    batch_losses = []
    # 使用tqdm创建进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f'Training File [{file_idx}/{total_files}]', leave=True)
    for batch_idx, batch in pbar:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                #check_patients_data_diease(batch)
                loss = model.compute_batch_loss(batch)
                scaled_loss = loss / gradient_accumulation_steps
            # 反向传播
            scaler.scale(scaled_loss).backward()
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
            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
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
def validate_model(model,val_data_paths,mappings,index_set,batch_size,death_df):
    model.eval()
    print("\nLoading and merging validation data...")
    combined_val_data = {}
    for val_path in val_data_paths:
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
            filtered_val_data = filter_valid_patients(val_data)
            combined_val_data.update(filtered_val_data)
    #combined_val_data = dict(list(combined_val_data.items())[:300])
    print(f"\nTotal number of validation patients: {len(combined_val_data)}")
    # 创建合并后的数据加载器
    val_dataset = PatientDataset_disease(combined_val_data, mappings, index_set,death_df)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_disease
    )
    val_loader = tqdm(val_loader, desc="Validating")

    disease_metrics, death_metrics = evaluate_model_disease(model, val_loader)
    metrics = calculate_disease_metrics(disease_metrics,death_metrics)
    print_metrics_summary(metrics)
    return metrics
def main(batch_size, num_epochs, learning_rate, gradient_accumulation_steps, patience):
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    directory_path = r"data/code_dict.parquet"
    code_dict = pd.read_parquet(directory_path)
    directory_path = r"data/death_record_v2.csv"
    death_df = pd.read_csv(directory_path)
    pretrained_model_patch = "时间正则_best_model_score_0.1412_epoch_1_20250127_234527.pt"

    train_disease_model(train_data_paths, val_data_paths, code_dict, mappings, batch_size, num_epochs, learning_rate,
                gradient_accumulation_steps, patience,pretrained_model_patch,death_df)
if __name__ == "__main__":
    main(batch_size=1, num_epochs=200, learning_rate=0.0001, gradient_accumulation_steps=16,
         patience=3)
