import torch
from torch import nn
from typing import Optional
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # ewai
        self.relu = nn.ReLU()
        self.preflag_linear = nn.Linear(768, 2, bias=False)
    def forward(self, x):
        x = x.view(-1, 7, 7, 768).permute(0,3,1,2)
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        # FIXME
        avg = self.avg_pool(x)
        x = self.preflag_linear(avg.squeeze(-1).squeeze(-1))
        return x

class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout=0.2, d_model=768, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        batch_size = input1.size()[1]
        # 改进dif_as_kv
        dif = input2 - input1
        output_1 = self.cross(input1, dif)  # (Q,K,V)
        output_2 = self.cross(input2, dif)  # (Q,K,V)

        return output_1, output_2

    def cross(self, input, dif):
        # 第一种 RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, dif, dif)  # (Q,K,V)
        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

class Pretrained_model(nn.Module):
    def __init__(self, decoder_mode, finetune_gpt2, img_feature_h, img_feature_w):
        super(Pretrained_model, self).__init__()

        self.decoder_mode = decoder_mode
        self.img_feature_h = img_feature_h
        self.img_feature_w = img_feature_w

        self.finetune_gpt2 = finetune_gpt2

        self.gpt_embedding_size = 768  # self.gpt.transformer.wte.weight.shape[1]

        self.d_model = self.gpt_embedding_size
        # position embedding：
        l = 50
        self.l_embedding = nn.Embedding(l, int(self.d_model))

        self.w_embedding = nn.Embedding(img_feature_w, int(self.d_model / 2))
        self.h_embedding = nn.Embedding(img_feature_h, int(self.d_model / 2))
        self.temporal_embedding = nn.Embedding(2, int(self.d_model))

        encoder_self_layer = nn.TransformerEncoderLayer(1 * self.d_model, nhead=8,
                                                        dim_feedforward=int(4 * self.d_model))
        self.transformer_encoder = nn.TransformerEncoder(encoder_self_layer, num_layers=2)

        encoder_self_layer_classifier = nn.TransformerEncoderLayer(2 * self.d_model, nhead=8,
                                                                   dim_feedforward=int(4 * self.d_model))
        self.transformer_encoder_classifier = nn.TransformerEncoder(encoder_self_layer_classifier, num_layers=3)

        decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=8, dim_feedforward=self.d_model * 2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 1)

        self.classifier_projection = nn.Linear(2 * self.d_model, 2)

        # cls_token
        scale = self.d_model ** -0.5
        self.class_embedding_classifier_changeflag = nn.Parameter(scale * torch.randn(1, 2 * self.d_model))
        # cls_token
        self.class_embedding_A = nn.Parameter(scale * torch.randn(1, self.d_model))
        self.class_embedding_B = nn.Parameter(scale * torch.randn(1, self.d_model))


        self.conv_dif = nn.Sequential(
            nn.Conv2d(self.d_model, int(self.d_model / 2), kernel_size=3),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(self.d_model / 2)),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1))
        )
        self.linear_dif = nn.Linear(int(self.d_model / 2), 2)

        self.pre_linear = nn.Linear(self.gpt_embedding_size, self.d_model)


        self.CBAM = CBAMLayer(768)
        self.CrossTransformer = nn.ModuleList([CrossTransformer(dropout=0.2, d_model=768, n_head=8) for i in range(2)])

    def position_embedding_1D_func(self, embedding_text):
        batch = embedding_text.shape[0]
        Len_feat = embedding_text.shape[1]

        pos_l = torch.arange(Len_feat, device=device).to(device)

        position_embedding = self.l_embedding(pos_l)

        position_embedding = position_embedding.unsqueeze(0).repeat(batch, 1, 1, 1)  # (batch, l, d_model)
        position_embedding = position_embedding.view(batch, -1, self.d_model)
        embedding_text = embedding_text + position_embedding  # NLD

        return embedding_text

    def position_embedding_2D_func(self, img_feat_A, img_feat_B):
        batch = img_feat_B.shape[0]
        Len_feat = img_feat_B.shape[1]
        h = int(math.sqrt(Len_feat))
        w = h
        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        position_embedding = position_embedding.unsqueeze(0).repeat(batch, 1, 1, 1)  # (batch, h, w, d_model)
        position_embedding = position_embedding.view(batch, -1, self.d_model)
        img_feat_A = img_feat_A + position_embedding  # NLD
        img_feat_B = img_feat_B + position_embedding  # NLD

        return img_feat_A, img_feat_B

    def Siamese_bridge_net(self, class_embedding, img_feat):
        conc_A = torch.cat(
            [class_embedding.unsqueeze(0).expand(img_feat.shape[0], *class_embedding.shape),
             img_feat], dim=1)
        conc_A = self.transformer_encoder(conc_A.permute(1, 0, 2)).permute(1, 0, 2)  # NLD
        cls_A = conc_A[:, 0, :]  # self.cls_projection(conc_A[:, 0, :])
        img_refine = conc_A[:, 1:, :]  # NLD
        return cls_A, img_refine

    def Classifier(self, clip_emb, img_feat):
        # img_feat = self.pre_linear(img_feat)
        img_feat_A = img_feat[:, 0, ...]  # (N,L,768)
        img_feat_B = img_feat[:, 1, ...]
        batch = img_feat_B.shape[0]
        Len_feat = img_feat_B.shape[1]
        h = int(math.sqrt(Len_feat))


        # # # 2D image position_embedding
        img_feat_A, img_feat_B = self.position_embedding_2D_func(img_feat_A, img_feat_B)  # NLD

        img_feat = img_feat_B - img_feat_A  # torch.abs(img_feat_B-img_feat_A)#torch.cat([img_feat_A, img_feat_B],dim=-1)
        _, img_feat = self.position_embedding_2D_func(img_feat_A, img_feat)  # NLD

        img_feat = torch.cat([img_feat_A, img_feat_B], dim=-1)
        conc_A = torch.cat(
            [self.class_embedding_classifier_changeflag.unsqueeze(0).expand(img_feat.shape[0],
                                                                            *self.class_embedding_classifier_changeflag.shape),
             img_feat], dim=1)

        conc_A = self.transformer_encoder_classifier(conc_A.permute(1, 0, 2)).permute(1, 0, 2)  # NLD
        changeflag = self.classifier_projection(conc_A[:, 0, :])  # self.cls_projection(conc_A[:, 0, :])

        return changeflag

    def forward(self, tokens, changeflag, area, ori_img, clip_emb, featuremap, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):

        # bridge Network
        changeflag = self.Classifier(clip_emb, featuremap)
        # classifier_pre_flag
        return changeflag

    def set_finetune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        # for p in self.gpt.parameters():
        #     p.requires_grad = fine_tune
        # for p in self.gpt.lm_head.parameters():
        #     p.requires_grad = True