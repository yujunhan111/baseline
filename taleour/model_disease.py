from DiseasePredictionNetwork import DiseaseSpecificClassifier
import torch
import torch.nn as nn
import random
from torchvision.ops import sigmoid_focal_loss
import numpy as np
from model import EHRModel
from disease_codes import DISEASE_CODES, disease_weights
from tool import get_current_demographic


# 添加适配器模块
class DemographicAdapter(nn.Module):
    """将低维demographic特征转换为高维特征的适配器模块"""

    def __init__(self, input_dim=17, output_dim=70):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        ).cuda()

    def forward(self, x):
        return self.projection(x)


# 添加LOS预测器
class LOSPredictor(nn.Module):
    def __init__(self, demo_dim, hist_dim):
        super().__init__()

        self.demo_encoder = nn.Sequential(
            nn.Linear(demo_dim, 8),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 + 128, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, demo_features, hist_repr):
        demo_encoded = self.demo_encoder(demo_features)
        hist_encoded = self.hist_encoder(hist_repr)
        combined = torch.cat([demo_encoded, hist_encoded], dim=1)
        return self.classifier(combined)


class DeathClassifier(nn.Module):
    def __init__(self, demo_dim, hist_dim):
        super().__init__()

        self.demo_encoder = nn.Sequential(
            nn.Linear(demo_dim, 8),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 + 128, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, demo_features, hist_repr):
        demo_encoded = self.demo_encoder(demo_features)
        hist_encoded = self.hist_encoder(hist_repr)
        combined = torch.cat([demo_encoded, hist_encoded], dim=1)
        return self.classifier(combined)


class DiseaseModel(nn.Module):
    def __init__(self, pretrained_path, code_dict, use_adapter=False, input_dim=17):
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

        # 获取当前模型的state_dict
        model_state_dict = self.pretrained_model.state_dict()

        # 过滤掉不匹配的weights
        filtered_state_dict = {}
        for k, v in new_state_dict.items():
            # 跳过code_prediction_network层，因为我们不需要
            if 'code_prediction_network' in k:
                continue

            # 检查其他参数的形状是否匹配
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"跳过不匹配的权重: {k}")

        # 加载过滤后的预训练权重
        self.pretrained_model.load_state_dict(filtered_state_dict, strict=False)

        print("\n成功加载预训练模型的编码和表示层权重，跳过了代码预测部分")

        # 添加适配器 (如果需要)
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = DemographicAdapter(input_dim, 70)

        # 添加疾病预测层
        self.disease_classifiers = nn.ModuleDict({
            disease_name: DiseaseSpecificClassifier(70, 768)
            for disease_name in DISEASE_CODES.keys()
        })

        # 添加死亡预测层
        self.death_classifier = DeathClassifier(70, 768)

        # 添加LOS预测层
        self.los_classifier = LOSPredictor(70, 768)

        # 定义损失函数权重
        self.los_weight = torch.tensor([30.0]).cuda()  # 可根据数据分布调整
        self.death_weight = torch.tensor([45.0]).cuda()  # 可根据数据分布调整

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
        total_los_loss = torch.tensor(0.0).cuda()

        disease_counts = {disease: 0 for disease in DISEASE_CODES.keys()}
        death_count = 0
        los_count = 0

        # 从batch字典中获取数据
        demographic_features = batch['demographic'].cuda()  # [batch_size, demo_dim]
        disease_data = batch['disease_data']  # batch_size个病人的疾病数据
        death_los_data = batch['death_los_data']  # 新字段,包含死亡和LOS数据
        print("demographic_features",demographic_features.shape)
        # 应用适配器(如果需要)
        if self.use_adapter and demographic_features.shape[1] != 70:
            demographic_features = self.adapter(demographic_features)

        # 遍历每个病人的数据
        for patient_idx in range(len(demographic_features)):
            demo_feat = demographic_features[patient_idx].unsqueeze(0)  # 获取单个病人的人口统计特征
            patient_diseases = disease_data[patient_idx]  # 获取单个病人的所有疾病数据
            patient_death_los = death_los_data[patient_idx]  # 获取单个病人的死亡和LOS数据

            # 处理疾病预测
            for disease_name, disease_info in patient_diseases.items():
                label = disease_info['label']

                # 跳过未定义的标签
                if label == -1:
                    continue

                # 获取该病人该疾病的历史数据
                hist_codes = disease_info['history_codes']
                hist_times = disease_info['history_times']
                event_time = disease_info['event_time']
                print("len(hist_codes)",len(hist_codes))
                print("len(hist_codes)",len(hist_times))
                if not hist_codes or not hist_times:
                    continue

                # 获取编码嵌入
                hist_embeddings, hist_times_tensor = self.get_code_embeddings(hist_codes, hist_times)
                print("hist_embeddings.shape",hist_embeddings.shape)
                print("hist_times_tensor.shape",hist_times_tensor.shape)
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
                    pred = self.disease_classifiers[disease_name](
                        demographic_current,
                        repr.unsqueeze(0)
                    )
                    # 使用疾病特定的权重计算loss
                    label_tensor = torch.tensor([[label]], dtype=torch.float32).cuda()
                    criterion = nn.BCEWithLogitsLoss(
                        pos_weight=disease_weights.get(disease_name, torch.tensor([30.0]).cuda()))
                    loss = criterion(pred, label_tensor)
                    # 累加loss并记录疾病计数
                    total_disease_loss += loss
                    disease_counts[disease_name] += 1

            # 处理死亡预测
            if 'death' in patient_death_los:
                death_info = patient_death_los['death']
                label = death_info['label']
                hist_codes = death_info['history_codes']
                hist_times = death_info['history_times']
                event_time = death_info['event_time']

                if label == -1 or not hist_codes or not hist_times:
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
                    pred = self.death_classifier(
                        demographic_current,
                        repr.unsqueeze(0)
                    )
                    # 计算死亡预测loss
                    label_tensor = torch.tensor([[label]], dtype=torch.float32).cuda()
                    criterion = nn.BCEWithLogitsLoss(pos_weight=self.death_weight)
                    loss = criterion(pred, label_tensor)
                    total_death_loss += loss
                    death_count += 1

            # 处理LOS预测
            if 'los_gt_7' in patient_death_los:
                los_info = patient_death_los['los_gt_7']
                label = los_info['label']
                hist_codes = los_info['history_codes']
                hist_times = los_info['history_times']
                event_time = los_info['event_time']

                if label == -1 or not hist_codes or not hist_times:
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
                    pred = self.los_classifier(
                        demographic_current,
                        repr.unsqueeze(0)
                    )
                    # 计算LOS预测loss
                    label_tensor = torch.tensor([[label]], dtype=torch.float32).cuda()
                    criterion = nn.BCEWithLogitsLoss(pos_weight=self.los_weight)
                    loss = criterion(pred, label_tensor)
                    total_los_loss += loss
                    los_count += 1

        # 计算平均损失
        valid_disease_predictions = sum(disease_counts.values())
        avg_disease_loss = total_disease_loss / valid_disease_predictions if valid_disease_predictions > 0 else torch.tensor(
            0.0).cuda()
        avg_death_loss = total_death_loss / death_count if death_count > 0 else torch.tensor(0.0).cuda()
        avg_los_loss = total_los_loss / los_count if los_count > 0 else torch.tensor(0.0).cuda()

        # 添加正则化损失
        reg_loss = self.pretrained_model.time_weight_net.get_reg_loss()

        # 打印各部分损失
        if valid_disease_predictions > 0:
            print(f"Disease Loss: {avg_disease_loss.item():.4f} (samples: {valid_disease_predictions})")
        if death_count > 0:
            print(f"Death Loss: {avg_death_loss.item():.4f} (samples: {death_count})")
        if los_count > 0:
            print(f"LOS Loss: {avg_los_loss.item():.4f} (samples: {los_count})")

        # 返回总loss
        return avg_disease_loss + avg_death_loss + avg_los_loss + 0.1 * reg_loss