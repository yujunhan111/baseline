from tool import get_history_codes
from torch.utils.data import Dataset, DataLoader
from filter_patients import filter_valid_patients
import numpy as np
import torch
import pandas as pd
from DiseasePredictionNetwork import DiseaseSpecificClassifier
from disease_codes import DISEASE_CODES
class PatientDataset_disease(Dataset):
    # 标签定义
    POSITIVE_LABEL = 0.95
    NEGATIVE_LABEL = 0.05
    UNDEFINED_LABEL = -1  # 特殊标记表示未定义样本
    disease_codes = DISEASE_CODES

    def __init__(self, data, mappings, index_set,death_df):
        self.patient_ids = list(data.keys())
        self.data = data
        self.mappings = mappings
        self.index_set = index_set
        self.max_indices = {
            'gender': max(v for v in mappings['gender'].values() if isinstance(v, (int, float))),
            'race': max(v for v in mappings['race'].values() if isinstance(v, (int, float))),
            'marital_status': max(v for v in mappings['marital_status'].values() if isinstance(v, (int, float))),
            'language': max(v for v in mappings['language'].values() if isinstance(v, (int, float)))
        }
        # 创建病人ID到死亡信息的映射
        self.death_info = {
            str(row['subject_id']): {
                'in_hosp_die': row['in_hosp_die'],
                'icu_die': row['icu_die'],
                'icu_24hour_die': row['icu_24hour_die']
            }
            for _, row in death_df.iterrows()
        }
        print(f"Dataset initialized with {len(self.patient_ids)} patients")
    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        try:
            patient_id = self.patient_ids[idx]
            patient_data = self.data[patient_id]
            # 处理demographic特征
            demo = patient_data['demographics']
            features = [1.0]
            features.append(np.log1p(float(demo['age'])))
            for feature in ['gender', 'race', 'marital_status', 'language']:
                feature_value = demo[feature]
                if pd.isna(feature_value):
                    feature_value = 'nan'
                feat_idx = self.mappings[feature].get(feature_value, self.mappings[feature]['nan'])
                one_hot = [0] * (self.max_indices[feature] + 1)
                one_hot[feat_idx] = 1
                features.extend(one_hot)
            demographic_features = torch.tensor(features, dtype=torch.float32)

            # 获取所有visits
            visits = []
            for event in patient_data['events']:
                if event['codes']:
                    visit_codes = [(code['code_index'], np.log1p(code['time'] / (24 * 7)))
                                   for code in event['codes']
                                   if code['code_index'] in self.index_set and not pd.isna(code['time'])]
                    if visit_codes:
                        visits.append(visit_codes)

            # 为每个疾病找出首次出现时间和对应的历史codes
            disease_data = {}
            for disease_name, disease_codes_list in self.disease_codes.items():
                disease_codes_set = set(disease_codes_list)
                occurrence_count = 0
                first_occurrence_time = None

                # 寻找疾病首次出现时间
                for visit in visits:
                    visit_codes_set = {code_idx for code_idx, _ in visit}
                    matching_codes = visit_codes_set & disease_codes_set

                    if matching_codes:
                        occurrence_count += 1
                        if first_occurrence_time is None:
                            first_occurrence_time = min(time for code_idx, time in visit
                                                        if code_idx in matching_codes)

                # 确定标签和收集历史codes
                if occurrence_count >= 2:  # 阳性病例
                    history = get_history_codes(visits, first_occurrence_time,self.index_set)
                    if history:
                        disease_data[disease_name] = {
                            'label': 0.95,
                            'event_time': first_occurrence_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        }
                elif occurrence_count == 0:  # 阴性病例
                    eval_time = visits[-1][0][1]  # 最后一次visit开始前的时间
                    history = get_history_codes(visits, eval_time,self.index_set)
                    if history:
                        disease_data[disease_name] = {
                            'label': 0.05,
                            'event_time': eval_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        }
                else:  # 出现一次的未定义病例
                    if first_occurrence_time is not None:
                        history = get_history_codes(visits, first_occurrence_time,self.index_set)
                        if history:
                            disease_data[disease_name]  =  {
                                'label': self.UNDEFINED_LABEL,
                                'event_time': first_occurrence_time,
                                'history_codes': history[0],
                                'history_times': history[1]
                            }

            # 获取死亡标签和历史数据
            death_data = {}
            if visits:
                last_visit = visits[-1]
                last_time = last_visit[-1][1]
                event_time = last_time + 0.001
                history = get_history_codes(visits, event_time, self.index_set)

                if history:
                    death_info = self.death_info.get(patient_id, {})

                    # 处理死亡标签
                    def get_death_label(value):
                        if value == 1:
                            return self.POSITIVE_LABEL
                        elif value == 0:
                            return self.NEGATIVE_LABEL
                        else:  # value == -1
                            return self.UNDEFINED_LABEL

                    death_data = {
                        'in_hosp_die': {
                            'label': get_death_label(death_info.get('in_hosp_die', -1)),
                            'event_time': event_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        },
                        'icu_die': {
                            'label': get_death_label(death_info.get('icu_die', -1)),
                            'event_time': event_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        },
                        'icu_24hour_die': {
                            'label': get_death_label(death_info.get('icu_24hour_die', -1)),
                            'event_time': event_time,
                            'history_codes': history[0],
                            'history_times': history[1]
                        }
                    }

            return {
                'demographic': demographic_features,
                'disease_data': disease_data,
                'death_labels': death_data
            }

        except Exception as e:
            print(f"Error loading patient {idx}: {str(e)}")
            raise e
