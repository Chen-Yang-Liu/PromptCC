
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
from thop import profile
from torchstat import stat
# import profile
# import stat
from torchsummaryX import summary
class mymodel(nn.Module):
    """
    CNN_Encoder.
    """

    def __init__(self, encoded_image_size=14, attention_method="ByPixel"):
        super(mymodel, self).__init__()
        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method

        # input_model = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-101
        checkpoint = torch.load("../models_checkpoint/data/1-times/RSICCformer_D/Simis_baseline/layer_3_1/BEST_checkpoint_resnet101_MCCFormers_diff_as_Q_trans.pth.tar",map_location='cpu')

        self.encoder_image = checkpoint['encoder_image']
        self.encoder_feat = checkpoint['encoder_feat']
        self.decoder = checkpoint['decoder']

    def forward(self, imgs):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        caps = torch.tensor([[300,    7,  200,  282,  114, 200,  168,   49,    7,  103, 400,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0]])
        caplens = torch.tensor([[52]])

        device = torch.device('cpu')
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        self.encoder_image = self.encoder_image.to(device)
        self.encoder_feat = self.encoder_feat.to(device)
        self.decoder = self.decoder.to(device)
        img_A = imgs.clone()
        img_B = imgs.clone()
        img_A = self.encoder_image(img_A)
        img_B = self.encoder_image(img_B)
        fused_feat = self.encoder_feat(img_A, img_B)
        scores, caps_sorted, decode_lengths, sort_ind = self.decoder(fused_feat, caps, caplens)
        # encoder_out = encoder_out.view(1, -1, 2048)
        # predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.decoder(encoder_out,caps, caplens)  # (batch_size, 2048, image_size/32, image_size/32)

        return scores





#计算方式
# total1 = sum([param.nelement() for param in encoder.parameters()])
# total2 = sum([param.nelement() for param in decoder.parameters()])
# model =mymodel()
# total = sum([param.nelement() for param in model.parameters()])
# # total = total1+total2
# print("Total params: %.2fM" % (total/1e6))

# #计算方式2
model =mymodel()
imgs = model(torch.randn(1,3,256,256))
# print(imgs.shape)
stat(model, (3, 256, 256))
# stat(model, (14,14,2048))

input1 = torch.randn(1,3,256,256)
flops1, parms1 =profile(model,inputs=(input1,))
print("flops=", flops1)
print("parms=", parms1)

# img = torch.randn(1,14,14,2048)
# caps = torch.randn(1,52)
# caplens = torch.randn(1,1)
# flops2,parms2 =profile(decoder,inputs=(img,caps,caplens,))
# print(">>>>>>>>>>>>>>>>>>>>>。。。。。。。。。")

# print("flops=",flops1+flops2)
# print("parms=",parms1+parms2)



#
# import torchvision.models as models
# import torch
# from ptflops import get_model_complexity_info
#
# # net = models.resnet50()
# net = mymodel()
# macs, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
#                                        print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
