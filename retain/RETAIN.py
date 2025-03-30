import torch
import torch.nn as nn
import torch.nn.functional as F
from disease_codes import DISEASE_CODES, disease_weights
from death_weight import death_los_weights


class RETAIN(nn.Module):
    def __init__(self, num_codes, hidden_dim, disease_names, demographic_dim, death_los_types=['death', 'los_gt_7'],
                 dropout=0.5):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim
        self.disease_names = disease_names
        self.death_los_types = death_los_types
        self.disease_codes = DISEASE_CODES

        # 基础嵌入层
        self.code_embedding = nn.Linear(num_codes, hidden_dim)

        # 单向RNN层 - 使用GRU而不是简单的RNN以获得更好的性能
        self.alpha_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 注意力层
        self.attention_alpha = nn.Linear(hidden_dim, 1)
        self.attention_beta = nn.Linear(hidden_dim, hidden_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + demographic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codes)  # 只预测疾病，不包括死亡和LOS
        )

        # 用于死亡和LOS预测的专门分类器
        self.death_los_classifier = nn.Sequential(
            nn.Linear(hidden_dim + demographic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 单一输出用于二分类任务
        )

        self.criterion = nn.BCELoss()

    def get_disease_probability(self, logits, disease_name):
        """获取疾病的预测概率"""
        disease_codes = self.disease_codes[disease_name]
        # 确保索引不超出范围
        valid_codes = [code - 1 for code in disease_codes if code - 1 < len(logits)]
        if not valid_codes:
            # 如果没有有效代码，返回一个默认概率
            return torch.tensor([0.5], device=logits.device)
        probs = torch.sigmoid(logits[torch.tensor(valid_codes, device=logits.device)])
        return torch.max(probs).unsqueeze(0)  # 返回最大概率作为疾病预测概率

    def get_death_los_probability(self, context, demographic, death_los_type):
        """获取死亡或住院时长的预测概率"""
        # 检查context维度，确保其为二维
        if len(context.shape) == 1:
            context = context.unsqueeze(0)

        # 确保demographic是一维的，然后扩展为二维
        if len(demographic.shape) >= 2:
            demographic = demographic.squeeze()

        combined = torch.cat([context, demographic.unsqueeze(0)], dim=1)
        logit = self.death_los_classifier(combined)
        return torch.sigmoid(logit)  # 返回概率，形状为[1, 1]

    def process_sequence(self, codes_list, device):
        """处理访问序列数据"""
        seq_tensor = torch.zeros(len(codes_list), self.num_codes, device=device)
        for i, codes in enumerate(codes_list):
            if codes:  # 确保codes不为空
                indices = torch.tensor([code - 1 for code in codes if code <= self.num_codes], device=device)
                if len(indices) > 0:  # 确保存在有效的索引
                    seq_tensor[i].scatter_(0, indices, 1.0)
        return seq_tensor

    def process_single_prediction(self, history_codes, demographic, device):
        """处理单个预测序列"""
        if not history_codes or len(history_codes) == 0:
            return None

        try:
            # 处理访问序列
            seq_tensor = self.process_sequence(history_codes, device)

            # 如果处理后的序列为空，则返回None
            if seq_tensor.size(0) == 0:
                return None

            embedded = self.code_embedding(seq_tensor)
            embedded = embedded.unsqueeze(0)  # 添加批次维度

            # RETAIN注意力机制
            alpha_output, _ = self.alpha_gru(embedded)
            beta_output, _ = self.beta_gru(embedded)

            # 计算访问级别的注意力权重alpha
            e_alpha = self.attention_alpha(alpha_output).squeeze(-1)
            alpha = torch.softmax(e_alpha, dim=1)

            # 计算变量级别的注意力权重beta
            beta = torch.tanh(self.attention_beta(beta_output))

            # 应用注意力权重
            weighted = embedded * beta
            weighted = weighted * alpha.unsqueeze(-1)

            # 聚合上下文向量
            context = torch.sum(weighted, dim=1)

            # 预测
            # 确保demographic是一维
            if len(demographic.shape) >= 2:
                demographic = demographic.squeeze()

            combined = torch.cat([context, demographic.unsqueeze(0)], dim=1)
            logits = self.classifier(combined).squeeze()

            # 确保返回的上下文向量是二维的[1, hidden_dim]
            if len(context.shape) == 1:
                context = context.unsqueeze(0)

            return logits, context

        except Exception as e:
            print(f"Error in sequence processing: {str(e)}")
            return None

    def forward(self, batch, disease_weights=None, death_los_weights=None, is_training=True):
        demographic = batch['demographic']
        disease_data = batch['disease_data']
        death_los_data = batch['death_los_data']
        batch_size = demographic.size(0)
        device = demographic.device

        if not is_training:
            # 初始化预测结果
            predictions = {}

            # 初始化疾病预测收集器
            for disease_name in self.disease_names:
                predictions[disease_name] = {'predictions': [], 'labels': []}

            # 初始化死亡和LOS预测收集器
            for death_los_type in self.death_los_types:
                predictions[death_los_type] = {'predictions': [], 'labels': []}

            # 遍历每个病人
            for b in range(batch_size):
                # 疾病预测
                for disease_name in self.disease_names:
                    if disease_name in disease_data[b] and disease_data[b][disease_name]['label'] != -1:
                        try:
                            result = self.process_single_prediction(
                                disease_data[b][disease_name]['history_codes'],
                                demographic[b],
                                device
                            )
                            if result is not None:
                                logits, context = result
                                prob = self.get_disease_probability(logits, disease_name)
                                predictions[disease_name]['predictions'].append(prob)
                                predictions[disease_name]['labels'].append(disease_data[b][disease_name]['label'])
                        except Exception as e:
                            print(f"Error processing disease sequence: {str(e)}")
                            continue

                # 死亡和LOS预测
                for death_los_type in self.death_los_types:
                    if death_los_type in death_los_data[b] and death_los_data[b][death_los_type]['label'] != -1:
                        try:
                            result = self.process_single_prediction(
                                death_los_data[b][death_los_type]['history_codes'],
                                demographic[b],
                                device
                            )
                            if result is not None:
                                logits, context = result
                                prob = self.get_death_los_probability(context, demographic[b], death_los_type)
                                predictions[death_los_type]['predictions'].append(prob)
                                predictions[death_los_type]['labels'].append(death_los_data[b][death_los_type]['label'])
                        except Exception as e:
                            print(f"Error processing death/LOS sequence: {str(e)}")
                            continue

            return predictions

        else:  # 训练模式
            total_loss = torch.tensor(0.0, device=device)
            total_weighted_samples = 0

            # 遍历每个病人
            for b in range(batch_size):
                # 处理疾病预测
                for disease_name in self.disease_names:
                    if disease_name in disease_data[b] and disease_data[b][disease_name]['label'] != -1:
                        try:
                            result = self.process_single_prediction(
                                disease_data[b][disease_name]['history_codes'],
                                demographic[b],
                                device
                            )
                            if result is not None:
                                logits, context = result
                                prob = self.get_disease_probability(logits, disease_name)
                                label = torch.tensor([disease_data[b][disease_name]['label']],
                                                     device=device,
                                                     dtype=torch.float)

                                # 确保prob和label形状匹配
                                if prob.shape != label.shape:
                                    prob = prob.reshape(label.shape)

                                # 获取疾病权重
                                weight = disease_weights.get(disease_name, 1.0) if disease_weights else 1.0

                                loss = self.criterion(prob, label)
                                total_loss += loss * weight
                                total_weighted_samples += 1
                        except Exception as e:
                            print(f"Error in disease training: {str(e)}")
                            continue

                # 处理死亡和LOS预测
                for death_los_type in self.death_los_types:
                    if death_los_type in death_los_data[b] and death_los_data[b][death_los_type]['label'] != -1:
                        try:
                            result = self.process_single_prediction(
                                death_los_data[b][death_los_type]['history_codes'],
                                demographic[b],
                                device
                            )
                            if result is not None:
                                logits, context = result
                                prob = self.get_death_los_probability(context, demographic[b], death_los_type)
                                label = torch.tensor([death_los_data[b][death_los_type]['label']],
                                                     device=device,
                                                     dtype=torch.float)

                                # 确保prob和label形状匹配
                                if prob.shape != label.shape:
                                    prob = prob.reshape(label.shape)

                                # 获取死亡/LOS权重
                                weight = 1.0
                                if death_los_weights and death_los_type in death_los_weights:
                                    weight = death_los_weights[death_los_type]

                                loss = self.criterion(prob, label)
                                total_loss += loss * weight
                                total_weighted_samples += 1
                        except Exception as e:
                            print(f"Error in death/LOS training: {str(e)}")
                            continue

            return total_loss / total_weighted_samples if total_weighted_samples > 0 else torch.tensor(0.0,
                                                                                                       device=device)