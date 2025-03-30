import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import EHRModel
from numpy.core.multiarray import scalar
from numpy import dtype
from evaluation import evaluate_model, compute_prediction_metrics, thinning_predict_next_time
from custom_collate import custom_collate
from get_patient_data import PatientDataset
from tqdm import tqdm
import numpy as np
from filter_patients import filter_valid_patients
from mappings import create_clean_mapping, create_all_mappings
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict


class EHRTester:
    def __init__(self, model_path, code_dict_path):
        """
        Initialize the tester

        Args:
            model_path: Path to model checkpoint
            code_dict_path: Path to diagnosis code dictionary
        """
        self.model_path = model_path
        self.code_dict = pd.read_parquet(code_dict_path)
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """Load the model"""
        self.model = EHRModel(self.code_dict, demo_dim=70)
        try:
            checkpoint = torch.load(
                self.model_path,
                weights_only=True,
                map_location=self.device
            )
        except Exception as e:
            print(f"Warning: Falling back to default loading. Error: {str(e)}")
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
            )

        print("\nBefore loading:")
        print("Time weight coefficients:", self.model.time_weight_net.coefficients.data)
        print("History Q_base:", self.model.history_repr.Q_base.data)

        # Process state dict
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Load state dict
        self.model.load_state_dict(new_state_dict)
        print("\nAfter loading:")
        print("Time weight coefficients:", self.model.time_weight_net.coefficients.data)
        print("History Q_base:", self.model.history_repr.Q_base.data)

        if self.device == 'cuda':
            self.model = self.model.cuda()
        self.model.eval()
        self.model = torch.compile(self.model, mode="reduce-overhead")
        print(f"\nModel successfully loaded from {self.model_path}")
        return checkpoint

    def evaluate_dataset(self, test_data, mappings, batch_size=1):
        """
        Evaluate the model on test dataset

        Args:
            test_data: Dictionary containing patient records
            mappings: Dictionary containing various code mappings
            batch_size: Batch size for evaluation
        """
        if self.model is None:
            self.load_model()

        index_set = set(self.code_dict["index"])
        dataset = PatientDataset(test_data, mappings, index_set, 2)

        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate
        )

        print("\nStarting evaluation...")
        k_values = [5, 10, 20, 30, 50, 100]
        test_loader = tqdm(test_loader, desc="Validating")
        results = evaluate_model(self.model, test_loader, k_values)

        # 打印评估结果
        self._print_evaluation_results(results, k_values)

        return results

    def _print_evaluation_results(self, results, k_values):
        """打印评估结果的辅助方法"""
        avg_metrics = results['metrics']

        print("\nEvaluation Results:")
        print("\nTime Prediction Metrics:")
        print(f"Mean Time Error: {avg_metrics['mean_time_error']:.2f}")
        print(f"Median Time Error: {avg_metrics['median_time_error']:.2f}")
        print(f"Mean Relative Error: {avg_metrics['mean_relative_error']:.2f}")
        print(f"Median Relative Error: {avg_metrics['median_relative_error']:.2f}")

        print("\nCode Prediction Metrics:")
        for k in k_values:
            print(f"\nMetrics for k={k}:")
            print(f"Precision@{k}: {avg_metrics[f'precision@{k}']:.4f}")
            print(f"Recall@{k}: {avg_metrics[f'recall@{k}']:.4f}")
            print(f"F1@{k}: {avg_metrics[f'f1@{k}']:.4f}")

        print(f"\nBest F1 Score: {avg_metrics['best_f1']:.4f} (k={avg_metrics['best_f1_k']})")


def main():
    # Set paths
    MODEL_PATH = 'best_model_score_-0.1647_epoch_1_20250121_225549.pt'
    CODE_DICT_PATH = r"data\code_dict.parquet"

    # Initialize tester
    tester = EHRTester(MODEL_PATH, CODE_DICT_PATH)

    # Load test data
    test_data = {}
    test_batches = [
        r"data\patient_records_batch_17.pkl",
        r"data\patient_records_batch_18.pkl",
        r"data\patient_records_batch_19.pkl",
        r"data\patient_records_batch_20.pkl"
    ]

    # 加载人口统计数据
    directory_path = r"data/patients_dict.csv"
    patient = pd.read_csv(directory_path)

    # Load and filter test data
    for batch_path in test_batches:
        with open(batch_path, 'rb') as f:
            batch_data = pickle.load(f)
            filtered_batch = filter_valid_patients(batch_data)
            test_data.update(filtered_batch)
    #test_data = dict(list(test_data.items())[:500])
    print(f"Total number of test patients: {len(test_data)}")

    # Create mappings
    mappings = create_all_mappings(patient)

    # Run evaluation
    results = tester.evaluate_dataset(test_data, mappings)


if __name__ == "__main__":
    main()