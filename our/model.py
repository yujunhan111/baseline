import torch
import torch.nn as nn
import random
from torchvision.ops import sigmoid_focal_loss
import numpy as np

from time_aware_attention import TimeAwareAttention, HistoryRepresentation, PolynomialTimeWeightNet
from IntensityNet import IntensityNet
from CodePredictionNetwork import CodePredictionNetwork
from tool import  get_current_demographic

class EHRModel(nn.Module):
    def __init__(self, code_dict, demo_dim):
        super().__init__()
        # 初始化时，将所有code embeddings转为字典存在GPU上
        self.embeddings = torch.tensor(code_dict['pubmedbert_embedding'].tolist(), dtype=torch.float32).cuda()
        self.index_to_embedding = {idx: emb for idx, emb in zip(code_dict['index'], self.embeddings)}

        embed_dim = len(code_dict['pubmedbert_embedding'][0])
        # TimeAwareAttention
        self.time_weight_net = PolynomialTimeWeightNet()
        self.attention = TimeAwareAttention(embed_dim, self.time_weight_net)
        self.history_repr = HistoryRepresentation(embed_dim, self.time_weight_net)
        self.intensity_net = IntensityNet(demo_dim, embed_dim)
        self.code_vocab_size = len(code_dict['index'])
        self.code_prediction_network = CodePredictionNetwork(
            demo_dim=demo_dim,
            hist_dim=embed_dim,
            num_codes=self.code_vocab_size
        )

    def compute_code_prediction_loss(self, code_probs, true_codes):
        """
        Args:
            code_probs: [1, num_codes] - 预测的概率
            true_codes: List[int] - 实际的code indices（从1开始）
        """
        true_labels = torch.full_like(code_probs, 0.05)  # [1, num_codes]
        true_labels[0, torch.tensor([c - 1 for c in true_codes])] = 0.95
        return sigmoid_focal_loss(
            code_probs,
            true_labels,
            alpha=0.25, gamma=2.0,
            reduction='mean'
        )

    def compute_least_squares_loss(self, true_visit_intensity, sampled_visit_intensity, all_vist_end):
        """
        计算最小二乘损失

        Args:
            true_visit_intensity (List[torch.Tensor]): 真实访问时间点的强度列表
            sampled_visit_intensity (List[torch.Tensor]): 采样时间点的强度列表
            all_vist_end : 病人所有visit结束时间

        Returns:
            torch.Tensor: least squares loss
        """
        intensity_norm = torch.zeros(1, 1).cuda()
        integral_term = torch.zeros(1, 1).cuda()

        # 处理采样点的强度平方和
        for intensity in sampled_visit_intensity:
            intensity_norm += intensity * intensity

        # 处理真实访问点的强度和
        for intensity in true_visit_intensity:
            integral_term += intensity

        # 归一化
        if sampled_visit_intensity:
            intensity_norm = intensity_norm / len(sampled_visit_intensity)
        integral_term = (2.0 / all_vist_end) * integral_term

        loss = intensity_norm - integral_term + 10
        return loss

    def get_code_embeddings(self, codes, times):
        """
        一次性获取所有codes的embeddings
        """
        indices = (torch.tensor(codes, dtype=torch.long) - 1).cuda()
        embeddings = self.embeddings[indices]
        times_tensor = torch.tensor(times, dtype=torch.float32).cuda()
        return embeddings, times_tensor

    def true_visit_repr(self, demographic, true_last_visit_end_times, accumulated_codes, accumulated_times, visit_end_indices,
                        true_next_codes):
        """
        处理单个病人的真实就诊数据
        """
        true_visit_intensity = []
        code_loss = 0
        num_visits = len(true_last_visit_end_times)

        # 一次性获取所有embeddings
        all_embeddings, all_times = self.get_code_embeddings(accumulated_codes, accumulated_times)

        # 处理每个真实访问时间点
        for i, (current_time, true_codes) in enumerate(zip(true_last_visit_end_times, true_next_codes)):
            # 直接切片获取到当前visit的历史
            end_idx = visit_end_indices[i]
            #print("end_idx",end_idx)
            hist_embeddings = all_embeddings[:end_idx]
            #print("len(hist_embeddings)",len(hist_embeddings))
            hist_times = all_times[:end_idx]
            #print("current_time",current_time)
            event_representations = self.attention(hist_embeddings, hist_embeddings, hist_times)
            repr = self.history_repr(event_representations, hist_times, current_time)

            if repr is not None:
                demographic_current = get_current_demographic(demographic, current_time)
                #print(demographic_current)
                intensity = self.intensity_net(demographic_current, repr.unsqueeze(0))
                code_probs = self.code_prediction_network(demographic_current, repr.unsqueeze(0))
                true_visit_intensity.append(intensity)
                code_loss += self.compute_code_prediction_loss(code_probs, true_codes)

        return code_loss / np.sqrt(num_visits), true_visit_intensity

    def sample_visit_repr(self, demographic, sampled_times, accumulated_codes, accumulated_times, sampled_end_indices):
        """
        处理单个病人的采样时间点数据
        """
        sampled_visit_intensity = []

        # 一次性获取所有embeddings
        all_embeddings, all_times = self.get_code_embeddings(accumulated_codes, accumulated_times)
        # print("sampled_times",sampled_times,"sampled_end_indices",sampled_end_indices)
        # 处理每个采样时间点
        for current_time, end_idx in zip(sampled_times, sampled_end_indices):
            # 直接切片获取历史
            hist_embeddings = all_embeddings[:end_idx]
            hist_times = all_times[:end_idx]

            event_representations = self.attention(hist_embeddings, hist_embeddings, hist_times)
            repr = self.history_repr(event_representations, hist_times, current_time)

            if repr is not None:
                demographic_current = get_current_demographic(demographic, current_time)
                #print("sample的：",demographic_current)
                intensity = self.intensity_net(demographic_current, repr.unsqueeze(0))
                sampled_visit_intensity.append(intensity)

        return sampled_visit_intensity

    def compute_batch(self, batch_data):
        """
        处理一个batch的数据
        Args:
            batch_data: Dict - 包含多个病人的数据的batch
        """
        batch_size = len(batch_data['true_last_visit_end_times'])
        batch_loss = 0
        batch_code_loss = 0
        batch_time_loss = 0

        # 处理每个病人的数据
        for i in range(batch_size):
            # 获取该病人的数据
            demographic = batch_data['demographic'][i].unsqueeze(0).cuda()

            # true visit
            code_loss, true_visit_intensity = self.true_visit_repr(
                demographic=demographic,
                true_last_visit_end_times=batch_data['true_last_visit_end_times'][i],
                accumulated_codes=batch_data['accumulated_codes'][i],
                accumulated_times=batch_data['accumulated_times'][i],
                visit_end_indices=batch_data['visit_end_indices'][i],
                true_next_codes=batch_data['true_next_codes'][i]
            )

            # sample visit
            sampled_visit_intensity = self.sample_visit_repr(
                demographic=demographic,
                sampled_times=batch_data['sampled_time_points'][i],
                accumulated_codes=batch_data['accumulated_codes'][i],
                accumulated_times=batch_data['accumulated_times'][i],
                sampled_end_indices=batch_data['sampled_end_indices'][i]
            )

            time_loss = self.compute_least_squares_loss(
                true_visit_intensity,
                sampled_visit_intensity,
                batch_data['all_visit_end'][i]
            )
            batch_loss += 5*code_loss + 0.2 * time_loss
            batch_code_loss += 5*code_loss
            batch_time_loss += 0.2 * time_loss
            reg_loss = self.time_weight_net.get_reg_loss()
            batch_loss += 0.2*reg_loss
            #print("5*code_loss:",5*code_loss,"0.2 * time_loss:",0.1 * time_loss,"reg_loss",0.5*reg_loss)
        return batch_loss / batch_size , batch_code_loss/batch_size, batch_time_loss/batch_size

