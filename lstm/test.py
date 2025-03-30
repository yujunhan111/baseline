import torch
import pandas as pd
import pickle
from RETAIN_EX import  RETAIN_EX
from get_patient_data_diease import PatientDataset_disease
from custom_collate_disease import custom_collate_disease
from torch.utils.data import DataLoader
from evaluation_disease import evaluate_epoch
from mappings import create_all_mappings
from filter_patients import filter_valid_patients
from disease_codes import DISEASE_CODES
import json
from pathlib import Path


def create_test_dataloader(dataset, batch_size=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_disease
    )


def load_model(model_path, code_dict, device):
    model = RETAIN_EX(
        num_codes=len(code_dict),
        hidden_dim=128,
        disease_names=list(DISEASE_CODES.keys()),
        demographic_dim=70
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    return model


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置路径
    MODEL_PATH = 'dipole_model_epoch_20.pt'  # 修改为你的模型路径
    TEST_DATA_PATHS = [rf"C:\Users\yujun\Desktop\data/patient_records_batch_{i}.pkl" for i in range(17, 21)]  # 批次17-20
    CODE_DICT_PATH = r'C:\Users\yujun\Desktop\data/random_code_dict.parquet'
    DEMOGRAPHIC_PATH = r'C:\Users\yujun\Desktop\data/patients_dict.csv'
    DEATH_RECORD_PATH = r'C:\Users\yujun\Desktop\data/death_record_v2.csv'

    # 加载数据
    print("Loading code dictionary...")
    code_dict = pd.read_parquet(CODE_DICT_PATH)

    print("Loading demographic data...")
    patient_data = pd.read_csv(DEMOGRAPHIC_PATH)
    mappings = create_all_mappings(patient_data)

    print("Loading death records...")
    death_df = pd.read_csv(DEATH_RECORD_PATH)

    # 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, code_dict, device)
    model.eval()

    # 加载测试数据
    print("Loading test data...")
    test_data = {}
    for test_path in TEST_DATA_PATHS:
        print(f"Loading {test_path}...")
        try:
            with open(test_path, 'rb') as f:
                current_data = pickle.load(f)
                test_data.update(current_data)
                print(f"Successfully loaded {len(current_data)} patients from {test_path}")
        except Exception as e:
            print(f"Error loading {test_path}: {str(e)}")

    # 过滤有效患者
    print("\nFiltering valid patients...")
    filtered_test_data = filter_valid_patients(test_data)
    print(f"Total patients after filtering: {len(filtered_test_data)} / {len(test_data)}")
    #filtered_test_data = dict(list(filtered_test_data.items())[:500])

    print(f"\nCreating test dataset with {len(filtered_test_data)} patients...")
    test_dataset = PatientDataset_disease(
        filtered_test_data,
        mappings,
        set(code_dict.index),
        death_df
    )

    test_loader = create_test_dataloader(test_dataset, batch_size=1)

    # 评估模型
    print("\nStarting evaluation...")
    results = evaluate_epoch(model, test_loader, device)

    # 打印结果
    print("\nTest Results:")
    print("\nDisease Metrics:")
    for disease, metrics in results['disease_metrics'].items():
        print(f"\n{disease}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):  # 只打印数值型指标
                print(f"{metric_name}: {value:.4f}")

    print("\nAverage Metrics:")
    for metric_name, value in results['average_metrics'].items():
        if isinstance(value, (int, float)):  # 只打印数值型指标
            print(f"{metric_name}: {value:.4f}")



if __name__ == "__main__":
    main()