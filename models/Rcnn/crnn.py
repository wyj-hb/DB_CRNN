import torch
import torch.nn as nn
from PIL import Image #导入PIL库
from torchinfo import summary
from torchvision import transforms
import dataset
from DiG.modeling_finetune import PatchEmbed
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    def forward(self, input):
        #output:（seq_len, batch, num_directions * hidden_size）
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, cfg, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        nh = cfg.CRNN.nh
        nc = cfg.CRNN.nc
        nclass = cfg.CRNN.nclass
        imgH = cfg.CRNN.imgH
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        # TODO 一共有7层,ks为每一层的卷积大小,ps为padding,ss为stride,nm为每层的feature map的通道数
        ks = [3, 3, 3, 3, 3, 3, 3,3]
        ps = [1, 1, 1, 1, 1, 1, 1,1]
        ss = [1, 1, 1, 1, 1, 1, 1,1]
        nm = [64, 128, 256, 256, 512, 512, 512,512]
        cnn = nn.Sequential()
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 1))) # 512x2x16
        convRelu(6, True)
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((2, 1)))
        # 512x1x16
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.adapter = None
    def encoder(self,x,bridge,rec_feat):
        bridge_feat = bridge(rec_feat)
    def forward(self, input,bridge,norm):
        input, tgt, tgt_lens, rec_feat = input
        #bridge 结构
        if bridge:
            bridge_feat = bridge(rec_feat)
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(0,2,1)
        x = norm(conv + bridge_feat)
        x = x.permute(1,0,2)
        preds = self.rnn(x)#LSTM的输入:(seq_len,batch_size,input_size)
        # if self.adapter:
        #     for adapter in self.adapter:
        #         preds = adapter(preds,0)
        return preds