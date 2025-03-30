from DiseasePredictionNetwork import DiseaseSpecificClassifier
import torch
import torch.nn as nn
import random
from torchvision.ops import sigmoid_focal_loss
import numpy as np
from model import EHRModel
from disease_codes import DISEASE_CODES,disease_weights
from death_weight import death_weights
from DeathClassifier import DeathSpecificClassifier
from tool import  get_current_demographic
class DiseaseModel(nn.Module):
    def __init__(self, pretrained_path,code_dict):
        super().__init__()
        # 1. 加载预训练的EHR模型
        self.pretrained_model = EHRModel(code_dict, demo_dim=70)
        checkpoint = torch.load(pretrained_path)
        # 处理编译后的权重问题
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('_orig_mod.'):
                new_key = k.replace('_orig_mod.', '')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        # 加载预训练权重
        self.pretrained_model.load_state_dict(new_state_dict)
        # 添加疾病预测层
        self.disease_classifiers = nn.ModuleDict({
            disease_name: DiseaseSpecificClassifier(70, 768)
            for disease_name in DISEASE_CODES.keys()
        })
        # 添加死亡预测层
        self.death_classifiers = nn.ModuleDict({
            death_type: DeathSpecificClassifier(70, 768)
            for death_type in death_weights.keys()
        })

    def freeze_pretrained(self):
        """冻结预训练模型的参数"""
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def partial_unfreeze(self):
        """解冻 time_weight_net 和 Q_base"""
        # 先冻结所有参数
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        # 解冻 time_weight_net
        self.pretrained_model.time_weight_net.coefficients.requires_grad = True
        # 解冻 Q_base
        self.pretrained_model.history_repr.Q_base.requires_grad = True
    def get_code_embeddings(self, codes, times):
        """
        一次性获取所有codes的embeddings
        """
        indices = (torch.tensor(codes, dtype=torch.long) - 1).cuda()
        embeddings = self.pretrained_model.embeddings[indices]
        times_tensor = torch.tensor(times, dtype=torch.float32).cuda()
        return embeddings, times_tensor

    def compute_batch_loss(self, batch):
        total_disease_loss = torch.tensor(0.0).cuda()
        total_death_loss = torch.tensor(0.0).cuda()
        disease_counts = {disease: 0 for disease in DISEASE_CODES.keys()}
        death_counts = {death_type: 0 for death_type in death_weights.keys()}

        # 从batch字典中直接获取数据
        demographic_features = batch['demographic'].cuda()  # [batch_size, demo_dim]
        disease_data = batch['disease_data']  # batch_size个病人的疾病数据
        death_labels = batch['death_labels']
        # 遍历每个病人的数据
        for patient_idx in range(len(demographic_features)):
            demo_feat = demographic_features[patient_idx].unsqueeze(0)  # 获取单个病人的人口统计特征
            patient_diseases = disease_data[patient_idx]  # 获取单个病人的所有疾病数据
            patient_death = death_labels[patient_idx]
            # 对每个疾病单独计算
            for disease_name, disease_info in patient_diseases.items():
                label = disease_info['label']

                # 跳过未定义的标签
                if label == -1:
                    continue

                # 获取该病人该疾病的历史数据
                hist_codes = disease_info['history_codes']
                hist_times = disease_info['history_times']
                event_time = disease_info['event_time']
                if not hist_codes or not hist_times:
                    print("跳过")
                    continue
                # 获取编码嵌入
                hist_embeddings, hist_times_tensor = self.get_code_embeddings(hist_codes, hist_times)

                # 获取表示
                event_representations = self.pretrained_model.attention(
                    hist_embeddings,
                    hist_embeddings,
                    hist_times_tensor
                )

                repr = self.pretrained_model.history_repr(
                    event_representations,
                    hist_times_tensor,
                    event_time
                )
                if repr is not None:
                    # 为每个疾病单独预测
                    demographic_current = get_current_demographic(demo_feat, event_time)
                    # print("demo_feat_orginal",demo_feat)
                    # print("demographic_current",demographic_current)
                    pred = self.disease_classifiers[disease_name](
                        demographic_current,
                        repr.unsqueeze(0)
                    )
                    # 使用疾病特定的权重计算loss
                    label_tensor = torch.tensor([[label]], dtype=torch.float32).cuda()
                    criterion = nn.BCEWithLogitsLoss(pos_weight=disease_weights[disease_name])
                    loss = criterion(pred, label_tensor)
                    # 累加loss并记录疾病计数
                    total_disease_loss += loss
                    disease_counts[disease_name] += 1

            for death_type, death_info in patient_death.items():
                label = death_info['label']
                hist_codes = death_info['history_codes']
                hist_times = death_info['history_times']
                event_time = death_info['event_time']
                if label == -1:
                    continue
                if not hist_codes or not hist_times:
                    continue
                # 获取编码嵌入
                hist_embeddings, hist_times_tensor = self.get_code_embeddings(hist_codes, hist_times)

                # 获取表示
                event_representations = self.pretrained_model.attention(
                    hist_embeddings,
                    hist_embeddings,
                    hist_times_tensor
                )

                repr = self.pretrained_model.history_repr(
                    event_representations,
                    hist_times_tensor,
                    event_time
                )

                if repr is not None:
                    demographic_current = get_current_demographic(demo_feat, event_time)
                    pred = self.death_classifiers[death_type](
                        demographic_current,
                        repr.unsqueeze(0)
                    )
                    # 使用死亡特定的权重计算loss
                    label_tensor = torch.tensor([[label]], dtype=torch.float32).cuda()
                    criterion = nn.BCEWithLogitsLoss(pos_weight=death_weights[death_type])
                    loss = criterion(pred, label_tensor)
                    total_death_loss += loss
                    death_counts[death_type] += 1

        valid_disease_predictions = sum(disease_counts.values())
        valid_death_predictions = sum(death_counts.values())

        avg_disease_loss = total_disease_loss / valid_disease_predictions if valid_disease_predictions > 0 else torch.tensor(
            0.0).cuda()
        avg_death_loss = total_death_loss / valid_death_predictions if valid_death_predictions > 0 else torch.tensor(
            0.0).cuda()
        reg_loss = self.pretrained_model.time_weight_net.get_reg_loss()
        # print("avg_disease_loss",avg_disease_loss)
        # print("avg_death_loss",avg_death_loss)
        # print("reg_loss",reg_loss)

        return avg_disease_loss + avg_death_loss+reg_loss