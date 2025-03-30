from tool import get_history_codes
from torch.utils.data import Dataset, DataLoader
from filter_patients import filter_valid_patients
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from disease_codes import DISEASE_CODES


class DemographicAdapter(nn.Module):
    """将低维demographic特征转换为高维特征的适配器模块"""

    def __init__(self, input_dim=17, output_dim=70):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.projection(x)


class PatientDataset_full(Dataset):
    POSITIVE_LABEL = 0.95
    NEGATIVE_LABEL = 0.05
    UNDEFINED_LABEL = -1
    disease_codes = DISEASE_CODES

    def __init__(self, data, mappings, index_set, death_los_df, adapter=None, pad_features=True):
        self.patient_ids = list(data.keys())
        self.data = data
        self.mappings = mappings
        self.index_set = index_set
        self.adapter = adapter  # 可选的特征适配器
        self.pad_features = pad_features  # 使用零填充方法

        # 加载死亡和LOS数据
        self.death_los_df = death_los_df
        # 将SUBJECT_ID转为字符串以便匹配
        self.death_los_df['SUBJECT_ID'] = self.death_los_df['SUBJECT_ID'].astype(str)
        # 创建索引以加速查找
        self.death_los_dict = dict(zip(self.death_los_df['SUBJECT_ID'],
                                       zip(self.death_los_df['LOS_GT_7'],
                                           self.death_los_df['EXPIRE_FLAG'])))

        self.max_indices = {
            'gender': max(v for v in mappings['gender'].values() if isinstance(v, (int, float))),
            'race': max(v for v in mappings['race'].values() if isinstance(v, (int, float))),
            'marital_status': max(v for v in mappings['marital_status'].values() if isinstance(v, (int, float))),
            'language': max(v for v in mappings['language'].values() if isinstance(v, (int, float)))
        }
        print(f"Dataset initialized with {len(self.patient_ids)} patients")

        # 计算特征维度
        self.feature_dim = 1 + 1  # 偏置项 + 年龄
        for feature in ['gender', 'race', 'marital_status', 'language']:
            self.feature_dim += self.max_indices[feature] + 1

    def __len__(self):
        return len(self.patient_ids)

    def extend_demographic_features(self, features, target_dim=70):
        """将demographic特征扩展到目标维度"""
        if self.adapter is not None:
            # 使用适配器模型转换特征
            return self.adapter(features.unsqueeze(0)).squeeze(0)
        elif self.pad_features:
            # 使用零填充方法
            padded_features = torch.zeros(target_dim, dtype=features.dtype)
            padded_features[:len(features)] = features
            return padded_features
        else:
            # 如果没有指定处理方法，返回原始特征
            return features

    def __getitem__(self, idx):
        try:
            patient_id = self.patient_ids[idx]
            patient_data = self.data[patient_id]

            # 获取人口统计学特征
            demo = patient_data['demographics']
            features = [1.0]  # 偏置项
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

            # 扩展特征到70维
            demographic_features = self.extend_demographic_features(demographic_features, target_dim=70)

            # 处理病人的visit数据
            visits = []
            for event in patient_data['events']:
                if event['codes']:
                    visit_codes = [(code['code_index'], np.log1p(code['time'] / (24 * 7)))
                                   for code in event['codes']
                                   if code['code_index'] in self.index_set and not pd.isna(code['time'])]
                    if visit_codes:
                        visits.append(visit_codes)
            visits = sorted(visits, key=lambda x: x[0][1])  # 确保按时间排序

            # 处理疾病数据
            disease_data = {}
            for disease_name, disease_codes_list in self.disease_codes.items():
                disease_codes_set = set(disease_codes_list)
                occurrence_count = 0
                first_disease_code_time = None

                for visit in visits:
                    visit_codes_with_time = [(code_idx, time) for code_idx, time in visit]
                    for code_idx, time in visit_codes_with_time:
                        if code_idx in disease_codes_set:
                            occurrence_count += 1
                            if first_disease_code_time is None:
                                first_disease_code_time = time
                            break

                if occurrence_count > 0:
                    history = get_history_codes(visits, first_disease_code_time, self.index_set, days_before=0)
                    if history:
                        disease_data[disease_name] = {
                            'label': self.POSITIVE_LABEL,
                            'event_time': first_disease_code_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        }
                elif occurrence_count == 0:
                    eval_time = visits[-1][-1][1] if visits and visits[-1] else 0
                    history = get_history_codes(visits, eval_time, self.index_set)
                    if history:
                        disease_data[disease_name] = {
                            'label': self.NEGATIVE_LABEL,
                            'event_time': eval_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        }

            # 从death_los数据获取死亡和LOS标签
            death_los_labels = {}
            if patient_id in self.death_los_dict:
                los_gt_7, expire_flag = self.death_los_dict[patient_id]

                # 准备死亡预测数据
                eval_time = visits[-1][-1][1] if visits and visits[-1] else 0
                history = get_history_codes(visits, eval_time, self.index_set)

                if history:
                    # 添加死亡标签
                    death_los_labels['death'] = {
                        'label': self.POSITIVE_LABEL if expire_flag == 1 else self.NEGATIVE_LABEL,
                        'event_time': eval_time,
                        'history_codes': history[0],
                        'history_times': history[1]
                    }

                    # 添加LOS标签
                    death_los_labels['los_gt_7'] = {
                        'label': self.POSITIVE_LABEL if los_gt_7 == 1 else self.NEGATIVE_LABEL,
                        'event_time': eval_time,
                        'history_codes': history[0],
                        'history_times': history[1]
                    }

            return {
                'demographic': demographic_features,
                'disease_data': disease_data,
                'death_los_data': death_los_labels
            }
        except Exception as e:
            print(f"Error loading patient {idx} (ID: {patient_id}): {str(e)}")
            raise e