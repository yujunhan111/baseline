import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_disease import DiseaseModel
from get_patient_data_diease import PatientDataset_disease
from custom_collate_disease import custom_collate_disease
from mappings import create_all_mappings
from evaluation_disease import evaluate_model_disease, calculate_disease_metrics, print_metrics_summary
from filter_patients import filter_valid_patients
from tool import plot_all_roc_curves

def model(test_batches, model_path, pretrained_path, batch_size=1):
    """
    在测试集上评估模型性能
    """
    print("\nLoading necessary data...")

    # 1. 加载代码字典
    #code_dict = pd.read_parquet("data/code_dict.parquet")
    code_dict = pd.read_parquet("data/code-dict-pubmedbert.parquet")

    # 2. 加载患者人口统计数据并创建映射
    patients = pd.read_csv("data/patients_dict.csv")
    mappings = create_all_mappings(patients)

    # 3. 加载死亡记录数据
    death_df = pd.read_csv("data/death_record_v2.csv")

    # 4. 创建和加载模型
    print("\nInitializing and loading model...")
    model = DiseaseModel(pretrained_path=pretrained_path, code_dict=code_dict)
    # 打印加载预训练后的权重
    print("\n=== After Loading Pretrained Weights ===")
    print("Time weight coefficients:", model.pretrained_model.time_weight_net.coefficients.data)
    print("History Q_base:", model.pretrained_model.history_repr.Q_base.data)
    print("\nDisease Classifier Sample Weights:")
    for disease_name, classifier in list(model.disease_classifiers.items())[:3]:
        # Access first layer weights through sequential modules
        first_layer = classifier.demo_encoder[0]
        print(f"{disease_name} demo encoder: {first_layer.weight.data[0, :5]}")
        first_layer = classifier.hist_encoder[0]
        print(f"{disease_name} hist encoder: {first_layer.weight.data[0, :5]}")

    print("\nDeath Classifier Sample Weights:")
    for death_type, classifier in list(model.death_classifiers.items())[:3]:
        first_layer = classifier.demo_encoder[0]
        print(f"{death_type} demo encoder: {first_layer.weight.data[0, :5]}")
        first_layer = classifier.hist_encoder[0]
        print(f"{death_type} hist encoder: {first_layer.weight.data[0, :5]}")

    checkpoint = torch.load(model_path)

    # 处理编译后的权重
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('_orig_mod.'):
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    # 打印加载微调模型后的权重
    print("\n=== After Loading Fine-tuned Weights ===")
    print("Time weight coefficients:", model.pretrained_model.time_weight_net.coefficients.data)
    print("History Q_base:", model.pretrained_model.history_repr.Q_base.data)
    print("\nDisease Classifier Sample Weights:")
    for disease_name, classifier in list(model.disease_classifiers.items())[:3]:
        # Access first layer weights through sequential modules
        first_layer = classifier.demo_encoder[0]
        print(f"{disease_name} demo encoder: {first_layer.weight.data[0, :5]}")
        first_layer = classifier.hist_encoder[0]
        print(f"{disease_name} hist encoder: {first_layer.weight.data[0, :5]}")

    print("\nDeath Classifier Sample Weights:")
    for death_type, classifier in list(model.death_classifiers.items())[:3]:
        first_layer = classifier.demo_encoder[0]
        print(f"{death_type} demo encoder: {first_layer.weight.data[0, :5]}")
        first_layer = classifier.hist_encoder[0]
        print(f"{death_type} hist encoder: {first_layer.weight.data[0, :5]}")
    model = model.cuda()
    model.eval()

    # 5. 加载并合并测试数据
    print("\nLoading and merging test data...")
    combined_test_data = {}
    for test_path in test_batches:
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
            filtered_test_data = filter_valid_patients(test_data)
            combined_test_data.update(filtered_test_data)
    #combined_test_data = dict(list(combined_test_data.items())[:5000])
    print(f"\nTotal number of test patients: {len(combined_test_data)}")

    # 6. 创建测试数据加载器
    index_set = set(code_dict["index"])
    test_dataset = PatientDataset_disease(combined_test_data, mappings, index_set, death_df)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_disease
    )
    test_loader = tqdm(test_loader, desc="Testing")

    # 评估模型并绘制ROC曲线
    print("\nEvaluating model...")
    with torch.cuda.amp.autocast():
        disease_metrics, death_metrics = evaluate_model_disease(model, test_loader)

    # 绘制ROC曲线
    print("\nPlotting ROC curves...")
    plot_all_roc_curves(disease_metrics, death_metrics)

    # 9. 计算和打印指标
    metrics = calculate_disease_metrics(disease_metrics, death_metrics)
    print("\n=== Test Results ===")
    print_metrics_summary(metrics)

    return metrics


if __name__ == "__main__":
    # 定义测试数据路径
    test_batches = [
        # r"data/patient_records_batch_1.pkl",
         r"data/patient_records_batch_2.pkl",
         r"data/patient_records_batch_11.pkl",
        r"data/patient_records_batch_18.pkl"
    ]

    # 模型路径
    model_path = "pubmed_model_epoch_6_score_2.6594.pt"
    pretrained_path = "pubmed_时间正则_best_model_score_0.1439_epoch_2_20250327_062003.pt"

    # 运行测试
    metrics = model(test_batches, model_path, pretrained_path)