from torch.utils.data import Dataset, DataLoader
from filter_patients import filter_valid_patients
import numpy as np
import torch
import pandas as pd
from tool import get_history_codes


class PatientDataset(Dataset):
    def __init__(self, data, mappings, index_set, sample_points):
        self.patient_ids = list(data.keys())
        self.data = data
        self.mappings = mappings
        self.index_set = index_set
        self.sample_points = sample_points
        print(f"Dataset initialized with {len(self.patient_ids)} patients")
        self.max_indices = {
            'gender': max(v for v in mappings['gender'].values() if isinstance(v, (int, float))),
            'race': max(v for v in mappings['race'].values() if isinstance(v, (int, float))),
            'marital_status': max(v for v in mappings['marital_status'].values() if isinstance(v, (int, float))),
            'language': max(v for v in mappings['language'].values() if isinstance(v, (int, float)))
        }

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        try:
            patient_id = self.patient_ids[idx]
            patient_data = self.data[patient_id]
            demo = patient_data['demographics']
            features = [1.0]
            # 处理demographic特征
            features.append(np.log1p(float(demo['age'])))
            for feature in ['gender', 'race', 'marital_status', 'language']:
                feature_value = demo[feature]
                if pd.isna(feature_value):
                    feature_value = 'nan'
                feat_idx = self.mappings[feature].get(feature_value, self.mappings[feature]['nan'])
                one_hot = [0] * (self.max_indices[feature] + 1)
                one_hot[feat_idx] = 1
                features.extend(one_hot)

            # 转换为tensor
            demographic_features = torch.tensor(features, dtype=torch.float32)

            # 处理visits
            visits = []
            for event in patient_data['events']:
                if event['codes']:
                    visit_codes = [(code['code_index'], np.log1p(code['time'] / (24 * 7)))
                                   for code in event['codes']
                                   if code['code_index'] in self.index_set and not pd.isna(code['time'])]
                    if visit_codes:
                        visits.append(visit_codes)

            # 从第二次visit开始,收集之前的所有codes
            accumulated_codes = []
            accumulated_times = []
            visit_end_indices = []  # 记录每次visit结束的位置

            # 收集第一次visit到倒数第二次visit的codes
            for i in range(len(visits) - 1):
                accumulated_codes.extend([code for code, _ in visits[i]])
                accumulated_times.extend([time for _, time in visits[i]])
                visit_end_indices.append(len(accumulated_codes))
            # 准备数据(第二次到最后一次visit的true_code,第一次到倒数第二次的visit end time)
            true_last_visit_end_times = []
            true_next_codes = []
            for i in range(len(visits) - 1):  # 循环到倒数第二次visit
                current_visit = visits[i]
                current_end_time = current_visit[-1][1]  # 当前visit的最后一个code的时间
                next_visit = visits[i + 1]
                true_last_visit_end_times.append(current_end_time)
                true_next_codes.append([code_idx for code_idx, _ in next_visit])

            # 采样点数据
            sampled_time_points = []
            sampled_end_indices = []  # 记录每个采样点应该使用的历史长度

            for i in range(1, len(visits)):
                prev_visit_end = visits[i - 1][-1][1]
                current_time = visits[i][0][1]
                sample_times = np.random.uniform(prev_visit_end, current_time, self.sample_points)

                sampled_time_points.extend(sample_times)
                # 每个采样点使用到该时间点之前的所有codes
                sampled_end_indices.extend([visit_end_indices[i - 1]] * self.sample_points)
            # 准备时间评估所需的数据 - 只保存最后两次就诊的时间
            last_visit_eval_time = None
            second_last_visit = visits[-2]
            last_visit = visits[-1]
            last_visit_eval_time = {
                'second_last_visit_end': second_last_visit[-1][1],  # 倒数第二次就诊的最后一个code的时间
                'last_visit_start': last_visit[0][1]  # 最后一次就诊的第一个code的时间
            }
            all_visit_end= last_visit[-1][1]
            return {
                'demographic': demographic_features,
                # 所有历史codes和times
                'accumulated_codes': accumulated_codes,
                'accumulated_times': accumulated_times,
                # 真实visit数据
                'true_last_visit_end_times': true_last_visit_end_times,
                'true_next_codes': true_next_codes,
                'visit_end_indices': visit_end_indices,
                # 采样点数据
                'sampled_time_points': sampled_time_points,
                'sampled_end_indices': sampled_end_indices,
                'last_visit_eval_time': last_visit_eval_time,
                "all_visit_end":all_visit_end
            }
        except Exception as e:
            print(f"Error loading patient {idx}: {str(e)}")
            raise e