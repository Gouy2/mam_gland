import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    交叉注意力模块，用于融合两种不同模态的特征
    """
    def __init__(self, dim_deep, dim_radiomics, num_heads=4, dropout=0.1):
        """
        初始化交叉注意力模块
        
        Args:
            dim_deep: 深度学习特征维度
            dim_radiomics: 影像组学特征维度
            num_heads: 注意力头数量
            dropout: Dropout比例
        """
        super(CrossAttention, self).__init__()
        
        # 确保维度可以被头数整除
        self.dim_deep = dim_deep
        self.dim_radiomics = dim_radiomics
        self.num_heads = num_heads
        self.head_dim = dim_deep // num_heads
        
        # 深度学习特征到影像组学特征的注意力
        self.query_deep = nn.Linear(dim_deep, dim_deep)
        self.key_radio = nn.Linear(dim_radiomics, dim_deep)
        self.value_radio = nn.Linear(dim_radiomics, dim_deep)
        
        # 影像组学特征到深度学习特征的注意力
        self.query_radio = nn.Linear(dim_radiomics, dim_radiomics)
        self.key_deep = nn.Linear(dim_deep, dim_radiomics)
        self.value_deep = nn.Linear(dim_deep, dim_radiomics)
        
        # 输出投影
        self.proj_deep = nn.Linear(dim_deep, dim_deep)
        self.proj_radio = nn.Linear(dim_radiomics, dim_radiomics)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm_deep1 = nn.LayerNorm(dim_deep)
        self.norm_deep2 = nn.LayerNorm(dim_deep)
        self.norm_radio1 = nn.LayerNorm(dim_radiomics)
        self.norm_radio2 = nn.LayerNorm(dim_radiomics)
        
        # 前馈网络
        self.ffn_deep = nn.Sequential(
            nn.Linear(dim_deep, dim_deep * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_deep * 4, dim_deep)
        )
        
        self.ffn_radio = nn.Sequential(
            nn.Linear(dim_radiomics, dim_radiomics * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_radiomics * 4, dim_radiomics)
        )
        
    def forward(self, deep_features, radiomics_features):
        """
        前向传播
        
        Args:
            deep_features: 深度学习特征 [batch_size, dim_deep]
            radiomics_features: 影像组学特征 [batch_size, dim_radiomics]
            
        Returns:
            fused_deep: 融合后的深度学习特征
            fused_radio: 融合后的影像组学特征
        """
        batch_size = deep_features.size(0)
        
        # 保存残差连接
        residual_deep = deep_features
        residual_radio = radiomics_features
        
        # 为多头注意力准备形状
        deep_features = deep_features.unsqueeze(1)  # [batch_size, 1, dim_deep]
        radiomics_features = radiomics_features.unsqueeze(1)  # [batch_size, 1, dim_radiomics]
        
        # ------- 深度学习特征关注影像组学特征 -------
        # 计算查询、键、值
        q_deep = self.query_deep(deep_features) 
        k_radio = self.key_radio(radiomics_features)
        v_radio = self.value_radio(radiomics_features)
        
        # 计算注意力分数
        attn_scores_deep = torch.matmul(q_deep, k_radio.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs_deep = F.softmax(attn_scores_deep, dim=-1)
        attn_probs_deep = self.dropout(attn_probs_deep)
        
        # 应用注意力权重
        deep_attended = torch.matmul(attn_probs_deep, v_radio)
        deep_attended = self.proj_deep(deep_attended)
        
        # 第一个残差连接和层归一化
        deep_attended = deep_attended.squeeze(1)  # [batch_size, dim_deep]
        deep_attended = self.norm_deep1(residual_deep + deep_attended)
        
        # 前馈网络
        deep_output = self.ffn_deep(deep_attended)
        
        # 第二个残差连接和层归一化
        fused_deep = self.norm_deep2(deep_attended + deep_output)
        
        # ------- 影像组学特征关注深度学习特征 -------
        # 计算查询、键、值
        q_radio = self.query_radio(radiomics_features)
        k_deep = self.key_deep(deep_features)
        v_deep = self.value_deep(deep_features)
        
        # 计算注意力分数
        attn_scores_radio = torch.matmul(q_radio, k_deep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs_radio = F.softmax(attn_scores_radio, dim=-1)
        attn_probs_radio = self.dropout(attn_probs_radio)
        
        # 应用注意力权重
        radio_attended = torch.matmul(attn_probs_radio, v_deep)
        radio_attended = self.proj_radio(radio_attended)
        
        # 第一个残差连接和层归一化
        radio_attended = radio_attended.squeeze(1)  # [batch_size, dim_radiomics]
        radio_attended = self.norm_radio1(residual_radio + radio_attended)
        
        # 前馈网络
        radio_output = self.ffn_radio(radio_attended)
        
        # 第二个残差连接和层归一化
        fused_radio = self.norm_radio2(radio_attended + radio_output)
        
        return fused_deep, fused_radio
        
    def get_attention_maps(self, deep_features, radiomics_features):
        """
        获取注意力图，用于可视化
        
        Args:
            deep_features: 深度学习特征
            radiomics_features: 影像组学特征
            
        Returns:
            deep_to_radio_attn: 深度特征关注影像组学的注意力图
            radio_to_deep_attn: 影像组学关注深度特征的注意力图
        """
        # 准备形状
        deep_features = deep_features.unsqueeze(1)
        radiomics_features = radiomics_features.unsqueeze(1)
        
        # 深度特征关注影像组学
        q_deep = self.query_deep(deep_features)
        k_radio = self.key_radio(radiomics_features)
        deep_to_radio_attn = torch.matmul(q_deep, k_radio.transpose(-2, -1)) / math.sqrt(self.head_dim)
        deep_to_radio_attn = F.softmax(deep_to_radio_attn, dim=-1)
        
        # 影像组学关注深度特征
        q_radio = self.query_radio(radiomics_features)
        k_deep = self.key_deep(deep_features)
        radio_to_deep_attn = torch.matmul(q_radio, k_deep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        radio_to_deep_attn = F.softmax(radio_to_deep_attn, dim=-1)
        
        return deep_to_radio_attn.squeeze(1), radio_to_deep_attn.squeeze(1) 