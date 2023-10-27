from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch import nn
import numpy as np
from torch.nn.init import xavier_uniform_
from typing import Optional
import math
import clip
from torch.nn.modules.container import ModuleList
import copy

# from load_clsmodel import device
from load_clsmodel import Pretrained_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout=0.5, d_model=768, n_head=4):
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

class Image_Encoder(nn.Module):

    def __init__(self, clip_model_type, len_change_emmbed, clip_feat_dim, h=7, w=7, gpt_dim=768, n_head=8, n_layers=3, prompt_len=10, uni_prompt_1_len=0):

        super(Image_Encoder, self).__init__()

        self.clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        self.clip_model = self.clip_model.to(dtype=torch.float32)
        self.clip_model_cls, preprocess_cls = clip.load('ViT-B/32', device=device, jit=False)
        self.prompt_len = prompt_len
        self.uni_prompt_1_len = uni_prompt_1_len
        d_model = gpt_dim
        self.d_model = gpt_dim
        self.clip_feat_dim = clip_feat_dim
        self.n_layers = n_layers
        print("CC_Transformer_encoderlayers=", n_layers)

        "describle the content"
        self.projection = nn.Linear(clip_feat_dim, d_model)

        self.concat_projection = nn.Linear(2*d_model, d_model)
        self.flag_projection = nn.Linear(d_model, 2)

        # FIXME：layers =
        encoder_self_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=int(4 * d_model))
        self.transformer_encoder = nn.TransformerEncoder(encoder_self_layer, num_layers=self.n_layers // 10 % 10)
        # FIXME：layers =
        encoder_self_layer_2feat = nn.TransformerEncoderLayer(2 * d_model, n_head, dim_feedforward=int(8 * d_model))
        self.trans_encoder_2feat = nn.TransformerEncoder(encoder_self_layer_2feat,
                                                         num_layers=(self.n_layers % 10))

        # a transformer for trying:
        self.transformer = nn.ModuleList([CrossTransformer(dropout=0.2, d_model=d_model, n_head=n_head) for i in range(3)])

        # position_embedding
        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.temporal_embedding = nn.Embedding(2, int(2*d_model))

        # cls_token
        scale = d_model ** -0.5
        self.class_embedding_A = nn.Parameter(scale * torch.randn(1, d_model))
        self.class_embedding_B = nn.Parameter(scale * torch.randn(1, d_model))

        # prompt
        self.prompt = nn.Parameter(scale * torch.randn(self.prompt_len, d_model), requires_grad=True)

        # self.change_proto
        self.change_proto = nn.Parameter(scale * torch.randn(len_change_emmbed, d_model), requires_grad=True)
        self.nochange_proto = nn.Parameter(scale * torch.randn(len_change_emmbed, d_model), requires_grad=True)

        #
        self.logit_scale = nn.Parameter(scale * torch.ones([]) * np.log(1 / 0.07))
        self.prefix_A = nn.Parameter(scale * torch.randn(1, 2*d_model), requires_grad=True)
        self.prefix_B = nn.Parameter(scale * torch.randn(1, 2*d_model), requires_grad=True)

        gpt2_type = 'gpt2'
        # gpt2_type = r'C:\Users\lcy\.cache\huggingface\hub\models--gpt2\snapshots\e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.gpt_encoderimg = GPT2LMHeadModel.from_pretrained(gpt2_type)

        # cls_model
        self.classification_module = Pretrained_model(decoder_mode='gpt2', finetune_gpt2=False,
                                         img_feature_h=7,
                                         img_feature_w=7)

        model_path = './checkpoints/classification_model/cls_model.pth.tar'
        checkpoint = torch.load(model_path, map_location=device)
        model = checkpoint['model_state_dict()']
        self.classification_module.load_state_dict(model)
        self.classification_module.eval()

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
        img_feat_A = img_feat_A + position_embedding   # NLD
        img_feat_B = img_feat_B + position_embedding  # NLD
        return img_feat_A, img_feat_B

    def temporal_embedding_func(self, img_refine_A, img_refine_B):
        # # temporal embedding
        batch = img_refine_B.shape[0]
        Len_feat = img_refine_B.shape[1]
        temporal = torch.arange(2, device=device).to(device)
        temporal_embed = self.temporal_embedding(temporal)
        temporal_embedding = temporal_embed.unsqueeze(1).repeat(1, Len_feat, 1) # (2,L,d_model)
        temporal_embedding = temporal_embedding.unsqueeze(0).repeat(batch, 1, 1, 1)  # (batch, 2,L,d_model)
        img_refine_A = img_refine_A + temporal_embedding[:, 0, ...]  # NLD
        img_refine_B = img_refine_B + temporal_embedding[:, 1, ...]

        return img_refine_A, img_refine_B  # NLD

    def changeflag2prompt(self, changeflag):
        batch = changeflag.shape[0]
        changefilter = changeflag.unsqueeze(-1).unsqueeze(-1)
        nochangefilter = 1 - changefilter
        change_proto = self.change_proto.unsqueeze(0).expand(batch, *self.change_proto.shape)
        nochange_proto = self.nochange_proto.unsqueeze(0).expand(batch, *self.change_proto.shape)
        change_proto_prompt = change_proto * changefilter + nochange_proto * nochangefilter
        return change_proto_prompt

    def Siamese_bridge_net(self, class_embedding, img_feat):
        conc_A = torch.cat(
            [class_embedding.unsqueeze(0).expand(img_feat.shape[0], *class_embedding.shape),
             img_feat], dim=1)
        conc_A = self.transformer_encoder(conc_A.permute(1, 0, 2)).permute(1, 0, 2)  # NLD
        cls_A = conc_A[:, 0, :]
        img_refine = conc_A[:, 1:, :]  # NLD
        return cls_A, img_refine

    def forward(self, changeflag, ori_img):
        img_A = ori_img[:, 0, ...]
        img_B = ori_img[:, 1, ...]
        clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
        clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)
        clip_emb_A, img_feat_A = clip_emb_A.to(dtype=torch.float32), img_feat_A.to(dtype=torch.float32)
        clip_emb_B, img_feat_B = clip_emb_B.to(dtype=torch.float32), img_feat_B.to(dtype=torch.float32)
        featuremap = torch.cat([img_feat_A.unsqueeze(1), img_feat_B.unsqueeze(1)], dim=1)

        # GT changeflag or preflag for training
        preflag = self.classification_module.Classifier(0, featuremap)
        # changeflag = torch.argmax(preflag, 1)

        if self.clip_feat_dim != self.d_model:
            img_feat_A = self.projection(img_feat_A)  # (N,L,768)
            img_feat_B = self.projection(img_feat_B)
        batch = img_feat_B.shape[0]
        Len_feat = img_feat_B.shape[1]

        # 2D image position_embedding
        img_feat_A, img_feat_B = self.position_embedding_2D_func(img_feat_A, img_feat_B)  # NLD

        # bridge Network
        cls_A, img_refine_A = self.Siamese_bridge_net(self.class_embedding_A, img_feat_A)
        cls_B, img_refine_B = self.Siamese_bridge_net(self.class_embedding_B, img_feat_B)
        # img_refine_A, img_refine_B = img_feat_A, img_feat_B
        dif = img_refine_B - img_refine_A
        img_refine_A = torch.cat([img_refine_A, dif], dim=-1)
        img_refine_B = torch.cat([img_refine_B, dif], dim=-1)
        img_refine_A = self.trans_encoder_2feat(img_refine_A.permute(1, 0, 2)).permute(1, 0, 2)
        img_refine_B = self.trans_encoder_2feat(img_refine_B.permute(1, 0, 2)).permute(1, 0, 2)
        # img_refine_A = self.concat_projection(img_refine_A)
        # img_refine_B = self.concat_projection(img_refine_B)
        # temporal encoding
        img_refine_A, img_refine_B = self.temporal_embedding_func(img_refine_A, img_refine_B)
        img_refine_A = self.concat_projection(img_refine_A)
        img_refine_B = self.concat_projection(img_refine_B)
        fusion_feat = torch.cat([img_refine_A, img_refine_B], dim=1)  # NLD

        # 1\\Auto Generate prompt
        # project two changeflag to different prompt
        change_proto_prompt = self.changeflag2prompt(changeflag)
        # unified prompt for captioning
        prompt = self.prompt.unsqueeze(0).expand(batch, *self.prompt.shape)
        uni_prompt_1 = prompt[:, :self.uni_prompt_1_len, ...]
        uni_prompt_2 = prompt[:, self.uni_prompt_1_len:, ...]
        # all prompt
        output = torch.cat([fusion_feat, uni_prompt_1, change_proto_prompt, uni_prompt_2], dim=1)  # NLD

        # 2\\hand craft prompt
        # hand_craft_prompt = torch.tensor(self.tokenizer.encode('Describe differences between images:'), dtype=torch.int64).to(device)
        # hand_craft_prompt = self.gpt_encoderimg.transformer.wte(
        #     hand_craft_prompt.unsqueeze(0).repeat(batch, 1))  # N,3,D
        #
        # output = torch.cat([fusion_feat, hand_craft_prompt], dim=1)

        # return Sim_cls_AB, pre_flag, output
        return 0, preflag, output


class LEVIR_CC_CaptionModel(nn.Module):
    def __init__(self, encoder_mode, decoder_mode, prompt_len, uni_prompt_1_len, len_change_emmbed,
                 img_feature_dim, img_feature_h, img_feature_w, num_layers):
        super(LEVIR_CC_CaptionModel, self).__init__()

        self.decoder_mode = decoder_mode
        self.img_feature_h = img_feature_h
        self.img_feature_w = img_feature_w

        if self.decoder_mode == 'gpt2':
            gpt2_type = 'gpt2'
            # gpt2_type = r'C:\Users\lcy\.cache\huggingface\hub\models--gpt2\snapshots\e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
            self.gpt_decoder = GPT2LMHeadModel.from_pretrained(gpt2_type) #(lm_head): Linear(in_features=768, out_features=50257, bias=False)

        self.gpt_embedding_size = self.gpt_decoder.transformer.wte.weight.shape[1]

        self.ori_voc_size = self.gpt_decoder.lm_head.out_features
        self.lm_head_nochange = nn.Linear(self.gpt_embedding_size, self.ori_voc_size)
        self.lm_head_change = nn.Linear(self.gpt_embedding_size, self.ori_voc_size)
        self.gpt_decoder.lm_head = nn.Sequential()
        self.pred_flag_projection = nn.Linear(self.gpt_embedding_size, 2)

        self.Image_Encoder = Image_Encoder(clip_model_type=encoder_mode,
                                           len_change_emmbed=len_change_emmbed, clip_feat_dim=img_feature_dim,
                                           h=img_feature_h, w=img_feature_w,
                                            gpt_dim=self.gpt_embedding_size,
                                            n_layers=num_layers, prompt_len=prompt_len, uni_prompt_1_len=uni_prompt_1_len)

        d_model = self.gpt_embedding_size
        encoder_self_layer = nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=int(2 * d_model))
        self.text_encoder = nn.TransformerEncoder(encoder_self_layer, num_layers=3)
        self.pos_emb = nn.Embedding(51, int(d_model))
        self.text_cls = nn.Parameter(torch.randn(1, d_model), requires_grad=True)

        self.gpt_decoder.eval()
        self.Image_Encoder.clip_model.eval()
        self.Image_Encoder.classification_module.eval()

    def dual_branch_func(self, changeflag, out):
        output = self.lm_head_change(out)
        return output, 0

    def forward(self, tokens, changeflag, ori_img, mask: Optional[torch.Tensor] = None):

        Sim_cls_AB, pre_flag, prefix_projections = self.Image_Encoder(changeflag, ori_img) #.view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_text = self.gpt_decoder.transformer.wte(tokens)  # -> NLD

        loss = 0

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if self.decoder_mode == 'gpt2':
            out = self.gpt_decoder(inputs_embeds=embedding_cat, attention_mask=mask) #1
            # out = self.gpt_decoder(inputs_embeds=embedding_cat) #
            out = out.logits
            output, pre = self.dual_branch_func(changeflag, out)

        return loss, pre_flag, output


    def set_finetune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        # encoder
        for p in self.Image_Encoder.clip_model.parameters():
            p.requires_grad = False
        for p in self.Image_Encoder.classification_module.parameters():
            p.requires_grad = False
        # decoder
        for p in self.gpt_decoder.parameters():
            p.requires_grad = fine_tune
        for p in self.gpt_decoder.lm_head.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        super(LEVIR_CC_CaptionModel, self).train(mode)
        self.gpt_decoder.eval()
        self.Image_Encoder.clip_model.eval()
        self.Image_Encoder.classification_module.eval()
        return self
