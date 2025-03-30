import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#计算时间的weight的神经网络
class PolynomialTimeWeightNet(nn.Module):
    def __init__(self, degree=5, lambda_reg=0.02):
        super().__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(torch.randn(degree + 1))
        self.lambda_reg = lambda_reg

    def forward(self, time_diff):
        time_diff = torch.abs(time_diff)
        with torch.no_grad():
            self.coefficients.data.clamp_(-10, 10)
        weights = self.coefficients[-1]
        for i in range(self.degree - 1, -1, -1):
            weights = self.coefficients[i] + weights * time_diff
        return torch.sigmoid(weights)

    def get_reg_loss(self):
        high_order_coeff = self.coefficients[2:]
        return self.lambda_reg * torch.norm(high_order_coeff)
# class TimeWeightNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.time_encoder = nn.Sequential(
#             nn.Linear(1, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1),
#             nn.Sigmoid()
#         ).cuda()
#
#     def forward(self, time_diff):
#         time_diff = torch.abs(time_diff)
#         weight = self.time_encoder(time_diff)
#         return weight

#计算ej(t)
class TimeAwareAttention(nn.Module):
    def __init__(self, embed_dim, time_weight_net):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_weight_net = time_weight_net

    def forward(self, query_code_embeds, hist_embeddings, hist_times):
        """
        Args:
            query_code_embeds: [n_history, embed_dim] - 历史codes的查询embeddings
            hist_embeddings: [n_history, embed_dim] - 历史codes的键值embeddings
            hist_times: [n_history] - 历史times
        Returns:
            outputs: [n_history, embed_dim] - 所有codes的表示
        """
        # 1. 计算时间差矩阵
        times_expanded_i = hist_times.unsqueeze(1)  # [n_history, 1]
        times_expanded_j = hist_times.unsqueeze(0)  # [1, n_history]
        time_diffs = torch.abs(times_expanded_i - times_expanded_j)  # [n_history, n_history]

        # 2. 计算时间权重矩阵
        time_diffs = time_diffs.unsqueeze(-1)  # [n_history, n_history, 1]
        time_weights = self.time_weight_net(time_diffs)  # [n_history, n_history, 1]
        time_weights = time_weights.squeeze(-1)  # [n_history, n_history]

        # 3. 计算attention scores
        # [n_history, embed_dim] x [embed_dim, n_history] = [n_history, n_history]
        attn_scores = torch.matmul(query_code_embeds, hist_embeddings.transpose(0, 1))
        attn_scores = attn_scores / math.sqrt(self.embed_dim)  # [n_history, n_history]

        # 4. 应用时间权重
        attn_scores = attn_scores * time_weights  # [n_history, n_history]

        # 5. Softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # [n_history, n_history]

        # 6. 加权求和
        # [n_history, n_history] x [n_history, embed_dim] = [n_history, embed_dim]
        outputs = torch.matmul(attn_weights, hist_embeddings)
        return outputs  # [n_history, embed_dim]

#计算 h(t)
class HistoryRepresentation(nn.Module):
    def __init__(self, embed_dim, time_weight_net):
        super().__init__()
        self.Q_base = nn.Parameter(torch.randn(embed_dim))
        self.time_weight_net = time_weight_net
        # print(embed_dim)

    def forward(self, event_embeddings, event_times, current_time):
        """
        Args:
            event_embeddings: [n_history, embed_dim] - 所有历史事件的表示e_j(t)
            event_times: [n_history] - 历史事件发生的时间
            current_time: float - 当前时间点t
        Returns:
            h_t: [embed_dim] - 整合后的历史表示
        """
        # 1. 计算时间差|t-t_j|
        time_diffs = torch.abs(current_time - event_times).unsqueeze(-1)  # [n_history, 1]

        # 2. 计算时间权重 w(|t-t_j|)
        time_weights = self.time_weight_net(time_diffs) * 1  # [n_history, 1]
        # print("time_weights",time_weights)
        # 3. 计算注意力分数 Q_base^T * e_j(t)
        # Q_base [embed_dim] × event_embeddings [n_history, embed_dim]ᵀ = [n_history]
        attn_scores = torch.matmul(self.Q_base, event_embeddings.transpose(0, 1))

        # 4. 应用时间权重 (Q_base^T * e_j(t)) * w(|t-t_j|)
        attn_scores = attn_scores * time_weights.squeeze(-1)  # [n_history]
        # print("attn_scores(Q_base^T * e_j(t)) * w(|t-t_j|)",attn_scores)
        # 5. Softmax得到权重α_j
        alpha = F.softmax(attn_scores, dim=0)  # [n_history]
        # print("alpha",alpha)
        # 6. 加权求和得到h_t = Σ(α_j * e_j(t))
        h_t = torch.sum(alpha.unsqueeze(-1) * event_embeddings, dim=0)  # [embed_dim]
        # print("current_time:",current_time,"。h_t",h_t)
        # print("time_diffs",time_diffs[0])
        # print("time_weights",time_weights[0])
        return h_t