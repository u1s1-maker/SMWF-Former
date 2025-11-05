import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer,ProbAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np

from layers.Invertible import RevIN
from layers.LiftingScheme_Structured import LiftingSchemeStructuredMultiScale, InverseLiftingSchemeStructuredMultiScale

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


class moving_avg(nn.Module):


    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # x - B, C, L
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        # print(front.shape, x.shape, end.shape)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x





class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride) if not imputation else moving_avg_imputation(
            self.kernel_size, self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AdpWaveletBlock(nn.Module):

    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = getattr(configs, 'regu_details', 0.0)
        self.regu_approx = getattr(configs, 'regu_approx', 0.0)

        # === 方向一: 使用结构化多尺度 Lifting ===
        self.wavelet = LiftingSchemeStructuredMultiScale(
            configs.enc_in, input_size,
            k_size=configs.lifting_kernel_size,
            dilations=configs.lifting_dilations
        )
        # =========================================

        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):

        c, d = self.wavelet(x)
        x_new = c # 近似分量


        r = 0.0
        if self.regu_details > 0.0:
            # L1 稀疏性约束
            r += self.regu_details * torch.mean(torch.abs(d))
        if self.regu_approx > 0.0:
            # L2 稳定性约束 (均值距离)
            r += self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)

        x_new = self.norm_x(x_new)
        d = self.norm_d(d)


        return x_new, r, d


class InverseAdpWaveletBlock(nn.Module):
    """使用结构化多尺度逆 Lifting 的自适应逆小波块。"""
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()


        self.inverse_wavelet = InverseLiftingSchemeStructuredMultiScale(
            configs.enc_in, input_size, # input_size 是 c/d 的长度
            k_size=configs.lifting_kernel_size,
            dilations=configs.lifting_dilations
        )
        # =========================================

    def forward(self, c, d):

        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed



class ConvolutionalModule(nn.Module):

    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.depthwise_conv = nn.Conv1d(d_model * 2, d_model * 2, kernel_size=kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=d_model * 2, bias=True)
        self.glu = nn.GLU(dim=1)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, L, D]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2) # [B, D, L]
        x = self.pointwise_conv1(x) # [B, 2D, L]
        x = self.depthwise_conv(x)
        x = self.glu(x) # [B, D, L]
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2) # [B, L, D]
        return residual + x

class EnhancedEncoderLayer(nn.Module):


    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu",
                  factor=3, output_attention=False,
                 use_conv_module=True, conv_kernel_size=3):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.use_conv_module = use_conv_module
        attention = FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention)
        self.attention = AttentionLayer(attention, d_model, n_heads)


        if use_conv_module:
            self.conv_module = ConvolutionalModule(d_model, conv_kernel_size, dropout)


        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        # --- Layer Norms ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if use_conv_module:
            self.norm3 = nn.LayerNorm(d_model) # Conv 后也加 Norm

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # 1. Multi-Head Attention
        attn_output, visual_attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 2. Optional Convolutional Module
        if self.use_conv_module:
            conv_output = self.conv_module(x)
            x = x + self.dropout(conv_output)
            x = self.norm3(x)

        # 3. Feed Forward
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x, visual_attn


class EnhancedHierarchicalEncoder(nn.Module):

    def __init__(self, num_layers, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu",
                  factor=3, output_attention=False,
                 use_conv_module=True, conv_kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            EnhancedEncoderLayer(
                d_model, n_heads, d_ff, dropout, activation,
                factor, output_attention,
                use_conv_module, conv_kernel_size
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            if attn is not None:
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)


        if self.layers[0].attention.inner_attention.output_attention:
             return x, attns
        else:
             return x, None
class AttentiveChannelPool(nn.Module):

    def __init__(self, num_channels, d_model, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        hidden_dim = max(1, num_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels, bias=False), # 输出通道权重
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] == 0: return torch.zeros(x.shape[0], x.shape[2], device=x.device) # 处理空输入

        weights = self.fc(x.mean(dim=1))

        b, c, d = x.shape
        y = self.avg_pool(x.transpose(1,2)).view(b, d) # [B, D] 全局平均池化
        channel_scores = torch.mean(x, dim=-1) # [B, C] 使用特征均值作为分数基础
        channel_weights = F.softmax(channel_scores, dim=1).unsqueeze(-1) # [B, C, 1]

        # --- 加权池化 ---
        context = torch.sum(x * channel_weights, dim=1) # [B, D]
        return context


class GatedResidualConnection(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.gate_linear = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual_input):

        combined = torch.cat((x, residual_input), dim=-1)
        gate = torch.sigmoid(self.gate_linear(combined))
        output = self.layer_norm(residual_input + gate * self.dropout(x))
        return output

class EnhancedCrossScaleFusionLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu", use_gating=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.use_gating = use_gating

        # Cross-Attention (Q from enc_out, K/V from cross_kv)
        self.cross_attention = AttentionLayer(
            FullAttention(False, 3, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        # Gating (Optional)
        if use_gating:
            self.attn_gate = GatedResidualConnection(d_model, dropout)
            self.ffn_gate = GatedResidualConnection(d_model, dropout)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)


    def forward(self, query, kv, attn_mask=None):
        # 1. Cross Attention
        attn_out, _ = self.cross_attention(query, kv, kv, attn_mask=attn_mask)

        # 2. Gated Add & Norm or Standard Add & Norm
        if self.use_gating:
            query = self.attn_gate(attn_out, query)
        else:
            query = self.norm1(query + self.dropout(attn_out))

        # 3. Feed Forward
        ffn_out = self.ffn(query)

        # 4. Gated Add & Norm or Standard Add & Norm
        if self.use_gating:
            query = self.ffn_gate(ffn_out, query)
        else:
            query = self.norm2(query + self.dropout(ffn_out))

        return query


class EnhancedCrossScaleFusion(nn.Module):

    def __init__(self, num_layers, lifting_levels, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu",
                 context_pooling='mean', # 'mean' or 'attentive'
                 use_gating=True):
        super().__init__()
        self.lifting_levels = lifting_levels
        self.context_pooling = context_pooling


        if context_pooling == 'attentive':
            self.channel_attention = nn.Sequential(
                nn.Linear(d_model, 1), # 为每个通道生成一个分数
                nn.Sigmoid()
            )
            print("Using Attentive Channel Pooling for Cross-Scale Context.")
        else:
            print("Using Mean Channel Pooling for Cross-Scale Context.")


        self.fusion_layers = nn.ModuleList([
            EnhancedCrossScaleFusionLayer(
                d_model, n_heads, d_ff, dropout, activation, use_gating
            ) for _ in range(num_layers) # num_layers 控制融合深度
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out, coef_embedding_levels, detail_embed_proj_layers):

        cross_kv_list = []
        for i in range(self.lifting_levels):

            if i >= len(coef_embedding_levels) or coef_embedding_levels[i].numel() == 0 or \
               i >= len(detail_embed_proj_layers) or isinstance(detail_embed_proj_layers[i], nn.Identity):
                print(f"Warning: Skipping context generation for level {i} due to invalid input/projection.")
                continue

            detail_proj_embed = coef_embedding_levels[i] # [B, C, L_detail_i]

            B, C, L_detail = detail_proj_embed.shape
            detail_reshaped = detail_proj_embed.view(B * C, L_detail)
            projected_details = detail_embed_proj_layers[i](detail_reshaped).view(B, C, -1) # [B, C, D]
            D_model = projected_details.shape[-1] # 获取实际 D


            if self.context_pooling == 'attentive':
                # projected_details: [B, C, D]
                # 计算通道权重
                channel_scores = torch.mean(projected_details, dim=-1) # [B, C]
                channel_weights = F.softmax(channel_scores, dim=1).unsqueeze(-1) # [B, C, 1]
                # 或者使用 self.channel_attention
                # channel_weights = self.channel_attention(projected_details) # 输出 [B, C, 1] 或其他？需要确定 Attn 实现
                level_context = torch.sum(projected_details * channel_weights, dim=1) # [B, D]
            else: # Mean pooling
                level_context = projected_details.mean(dim=1)  # [B, D]

            cross_kv_list.append(level_context)


        if not cross_kv_list:
            print("Warning: No valid cross-scale context generated. Skipping fusion.")
            return enc_out # 直接返回原始输入


        cross_kv = torch.stack(cross_kv_list, dim=1)  # [B, num_valid_levels, D]


        fusion_query = enc_out
        for layer in self.fusion_layers:
            fusion_query = layer(fusion_query, cross_kv, attn_mask=None)

        return fusion_query
class ClusteredLinear(nn.Module):
    def __init__(self, n_clusters, enc_in, seq_len, pred_len, hidden_dim=32, activation='relu', dropout_rate=0.1): # 默认 hidden_dim=32, dropout=0.1
        super().__init__()
        self.n_clusters = n_clusters
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.trend_models = nn.ModuleDict()
        for cluster_id in range(n_clusters):
            layers = [
                nn.Linear(seq_len, hidden_dim),
            ]
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

            # <<< ADDED Dropout >>>
            layers.append(nn.Dropout(p=dropout_rate))

            layers.append(nn.Linear(hidden_dim, pred_len))

            self.trend_models[str(cluster_id)] = nn.Sequential(*layers)

    def forward(self, x, clusters):
        output_channels = []
        B, C, L = x.shape
        # <<< 保持 forward 逻辑不变 >>>
        assert C == self.enc_in
        assert C == len(clusters)
        assert L == self.seq_len

        for channel in range(C):
            cluster_id = str(clusters[channel].item())
            channel_data = x[:, channel, :].unsqueeze(1)
            transformed_channel = self.trend_models[cluster_id](channel_data)
            output_channels.append(transformed_channel)

        output = torch.cat(output_channels, dim=1)
        return output


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.lifting_levels = configs.lifting_levels
        self.kmeans = KMeans(n_clusters=configs.n_clusters)
        self.output_attention = configs.output_attention
        self.series_decomp = series_decomp(kernel_size=configs.moving_avg)
        #self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        # self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        trend_mlp_hidden_dim = configs.trend_mlp_hidden_dim
        trend_mlp_activation = configs.activation
        trend_mlp_dropout = configs.dropout

        self.trend_linear = ClusteredLinear(
            configs.n_clusters,
            configs.enc_in,
            self.seq_len,
            configs.pred_len,
            hidden_dim=trend_mlp_hidden_dim,
            activation=trend_mlp_activation,
            dropout_rate=trend_mlp_dropout
        )
        # Embedding
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        input_size = self.seq_len
        for i in range(self.lifting_levels):
            level_input_size = input_size
            self.encoder_levels.append(
                AdpWaveletBlock(configs, level_input_size)
            )
            input_size //= 2
            self.linear_levels.append(
                nn.Sequential(nn.Linear(input_size, input_size))
            )
            self.coef_linear_levels.append(
                nn.Sequential(nn.Linear(input_size, input_size))
            )
            # === MODIFICATION END ===
        self.input_size = input_size

        # 最低频分量的嵌入层
        self.enc_embedding = DataEmbedding_inverted(
            self.input_size, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        use_encoder_conv = getattr(configs, 'use_encoder_conv', True)
        encoder_conv_kernel = getattr(configs, 'encoder_conv_kernel', 3)

        self.encoder = EnhancedHierarchicalEncoder(
            num_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            factor=configs.factor,
            output_attention=configs.output_attention,
            use_conv_module=use_encoder_conv,
            conv_kernel_size=encoder_conv_kernel
        )
        # self.encoder = Encoder(
        #    [
        #        EncoderLayer(
        #            AttentionLayer(
        #                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                              output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #            configs.d_model,
        #            configs.d_ff,
        #            dropout=configs.dropout,
        #            activation=configs.activation
        #        ) for l in range(configs.e_layers)
        #    ],
        #    norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        self.use_cross_scale_attn = getattr(configs, 'use_cross_scale_attn', True)
        if self.use_cross_scale_attn:
            num_fusion_layers = getattr(configs, 'num_fusion_layers', 1)
            fusion_context_pooling = getattr(configs, 'fusion_context_pooling', 'attentive')
            use_fusion_gating = getattr(configs, 'use_fusion_gating', True)
            n_heads_cross = getattr(configs, 'n_heads_cross', configs.n_heads)
            self.cross_scale_fusion = EnhancedCrossScaleFusion(
                num_layers=num_fusion_layers,
                lifting_levels=configs.lifting_levels,
                d_model=configs.d_model,
                n_heads=n_heads_cross,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
                context_pooling=fusion_context_pooling,
                use_gating=use_fusion_gating

            )
            self.detail_embed_proj = nn.ModuleList()  # Correct init
            temp_size = self.seq_len
            print("--- Configuring Detail Embed Proj ---")
            for i in range(self.lifting_levels):
                detail_size_at_level = temp_size // 2
                print(f"  Level {i}: detail_size={detail_size_at_level}")
                self.detail_embed_proj.append(
                    nn.Linear(detail_size_at_level , configs.d_model)

                )

                temp_size //= 2
            self.cross_attn_norm = nn.LayerNorm(configs.d_model)
            self.cross_attn_dropout = nn.Dropout(configs.dropout)
        # =================================


        target_decode_start_len = self.input_size
        self.lowrank_projection = nn.Linear(configs.d_model, target_decode_start_len)


        self.decoder_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        for i in range(self.lifting_levels):
            current_level_size = self.input_size * (2 ** i)
            self.decoder_levels.append(
                InverseAdpWaveletBlock(configs, input_size=current_level_size)
            )
            self.coef_dec_levels.append(
                nn.Sequential(nn.Linear(current_level_size, current_level_size))
            )

        self.projection = nn.Linear(self.seq_len, self.pred_len)


        self.register_buffer('clusters', None)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters_input=None):

        B, L, C = x_enc.shape


        seasonal_init, trend_init = self.series_decomp(x_enc.permute(0, 2, 1))  # 输入 [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)  # [B, L, C]
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, L, C]

        trend_init_norm = self.rev_trend(trend_init, 'norm')  # [B, L, C]


        x_seasonal = seasonal_init.permute(0, 2, 1)  # [B, C, L]
        encoded_coefficients = []  # 存储原始细节分量
        x_embedding_levels = []  # 存储近似分量的投影
        coef_embedding_levels = []  # 存储细节分量的投影

        current_approx = x_seasonal

        for i in range(self.lifting_levels):

            approx_component, r, detail_component = self.encoder_levels[i](current_approx)
            encoded_coefficients.append(detail_component)
            # 应用投影层
            x_embedding_levels.append(self.linear_levels[i](approx_component))
            coef_embedding_levels.append(self.coef_linear_levels[i](detail_component))
            current_approx = approx_component

        final_approx_unprojected = current_approx  # [B, C, L_final]


        cluster_feature_source = trend_init.detach()
        cluster_input = cluster_feature_source.permute(2, 0, 1).reshape(C, B * L)
        #cluster_input = cluster_input.float()
        clusters_to_use = None
        # Determine which clusters to use
        if clusters_input is None:
            if self.clusters is None or self.clusters.shape[0] != C:
                if self.training:
                    with torch.no_grad():
                        clusters = self.kmeans.fit_predict(cluster_input)
                        self.clusters = clusters.to(x_enc.device)

                        clusters_to_use = self.clusters

                elif self.clusters is None:
                    with torch.no_grad():
                        clusters = self.kmeans.fit_predict(cluster_input)
                    clusters_to_use = clusters.to(x_enc.device)
                else:
                    clusters_to_use = self.clusters
            else:
                clusters_to_use = self.clusters
        else:
            clusters_to_use = clusters_input
        # === MODIFICATION END ===
        if clusters_to_use is None:
            raise RuntimeError("Failed to determine clusters for ClusteredLinear.")
        # --- Now predict trend using the computed/retrieved clusters_to_use ---
        trend_pred = self.trend_linear(trend_init_norm.permute(0, 2, 1), clusters_to_use)
        trend_pred = trend_pred.permute(0, 2, 1)  # [B, P, C]
        trend_pred = self.rev_trend(trend_pred, 'denorm')
        # ------------------------------


        enc_input = final_approx_unprojected.permute(0, 2, 1)
        enc_out = self.enc_embedding(enc_input, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [B, P, D]

        # === 方向二: 应用跨尺度注意力 ===
        if self.use_cross_scale_attn:
            cross_kv_list = []
            for i in range(self.lifting_levels):
                projected_for_attn = self.detail_embed_proj[i](detail_proj_embed)
                level_context = projected_for_attn.mean(dim=1)
                cross_kv_list.append(level_context)


            cross_kv = torch.stack(cross_kv_list, dim=1)  # [B, num_levels, D]


            fusion_output = self.cross_scale_fusion(
                enc_out,  # Query
                coef_embedding_levels,
                self.detail_embed_proj
            )


            enc_out = self.cross_attn_norm(enc_out + self.cross_attn_dropout(fusion_output))
        dec_start_proj = self.lowrank_projection(enc_out)

        if dec_start_proj.shape[1] == C:
            dec_start = dec_start_proj
        elif enc_out.shape[1] > 1:
            dec_start = dec_start_proj.mean(dim=1).unsqueeze(1).repeat(1, C, 1)
        else:
            dec_start = dec_start_proj.repeat(1, C, 1)

        current_reconstruction = dec_start  # Size L_final = 12

        approx_residuals_list = x_embedding_levels
        raw_details_list = encoded_coefficients
        coef_embedding_list = coef_embedding_levels

        for k in range(self.lifting_levels):  # k = 0, 1, 2
            dec_level = self.decoder_levels[k]
            c_linear_dec = self.coef_dec_levels[k]
            encoding_idx = self.lifting_levels - 1 - k


            approx_emb = approx_residuals_list[encoding_idx]
            details_original = raw_details_list[encoding_idx]
            coef_emb = coef_embedding_list[encoding_idx]


            dec_input = current_reconstruction

            if dec_input.shape == approx_emb.shape:
                dec_input = dec_input + approx_emb
                # print(f"Debug: Added approx residual k={k}")
            else:

                print(f"Warning: Skipping approx residual add at level k={k} due to shape mismatch: "
                      f"ReconIn {dec_input.shape}, ApproxEmb {approx_emb.shape}.")


            detail_input_for_inverse = None
            try:

                linear_layer = None
                if isinstance(c_linear_dec, nn.Sequential):
                    if len(c_linear_dec) > 0 and isinstance(c_linear_dec[0], nn.Linear):
                        linear_layer = c_linear_dec[0]
                elif isinstance(c_linear_dec, nn.Linear):
                    linear_layer = c_linear_dec

                if linear_layer is None:
                    raise TypeError(f"c_linear_dec at k={k} is not Linear or Sequential containing Linear.")
                if details_original.shape[-1] != linear_layer.in_features:
                    raise ValueError(
                        f"Shape mismatch for c_linear_dec input k={k}: expects {linear_layer.in_features}, got {details_original.shape[-1]}")

                transformed_details = c_linear_dec(details_original)

                if coef_emb.shape == transformed_details.shape:
                    detail_input_for_inverse = coef_emb + transformed_details
                    # print(f"Debug: Combined details k={k}")
                else:
                    print(
                        f"Warning: Shape mismatch for detail combination k={k}: CoefEmb {coef_emb.shape}, Transformed {transformed_details.shape}. Fallback.")
                    detail_input_for_inverse = details_original  # 回退
            except Exception as e:
                print(f"Error in complex detail processing k={k}: {e}. Fallback.")
                detail_input_for_inverse = details_original  # 回退


            if detail_input_for_inverse is None:
                detail_input_for_inverse = details_original


            current_reconstruction = dec_level(dec_input, detail_input_for_inverse)

        # --- Final projection ---
        seasonal_pred = self.projection(current_reconstruction)
        seasonal_pred = seasonal_pred.permute(0, 2, 1)

        # --- Combine Trend and Seasonal ---
        final_pred = seasonal_pred[:, -self.pred_len:, :] + trend_pred[:, -self.pred_len:, :]
        return final_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
     
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 调用 forecast，不传递 clusters_input，让 forecast 内部处理
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, clusters_input=None)
            return dec_out