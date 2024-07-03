import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from .segmentation_head import DBHead
from .segmentation_body import FPN
backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]}}
segmentation_body_dict = {'FPN': FPN}
segmentation_head_dict = {'DBHead': DBHead}
class DBTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained = True
        # TODO 配置adapter
        self.adapter = None
        backbone_name = "resnet18"
        segmentation_body_name = "FPN"
        segmentation_head_name = "DBHead"
        backbone_model, backbone_out = \
            backbone_dict[backbone_name]['models'], backbone_dict[backbone_name]['out']  # noqa
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_body = segmentation_body_dict[
            segmentation_body_name](backbone_out, inner_channels=256)
        self.segmentation_head = segmentation_head_dict[
            segmentation_head_name](self.segmentation_body.out_channels,
                                    out_channels=2)
        self.name = '{}_{}_{}'.format(backbone_name, segmentation_body_name,
                                      segmentation_head_name)
    def forward(self, x,istrain):
        """
        :return: TRAIN mode: prob_map, threshold_map, appro_binary_map
        :return: EVAL mode: prob_map, threshold_map
        """
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        if self.adapter:
            for adapter in self.adapter:
                segmentation_body_out = adapter(segmentation_body_out)
        segmentation_head_out = self.segmentation_head(segmentation_body_out,istrain)
        y = F.interpolate(segmentation_head_out,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True)
        return segmentation_body_out,y
if __name__ == '__main__':
    import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # model = DBTextModel().to('cpu')
    # data = torch.randn(1,3,224,224)
    # model(data)
    # summary(model,input_size = (1,3,224,224))
    # state_dict = torch.load('../models/db_resnet18.pth', map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # print(model)
    # # 假设模型是在训练过程中使用的标准化参数
    # # 如果不确定，可以查看模型训练时使用的数据预处理过程
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # # 图片预处理
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    #
    # # 加载一张图片（这里假设图片路径为 'path_to_your_image.jpg'）
    # image_path = '../assets/foo.jpg'
    # image = Image.open(image_path).convert('RGB')
    #
    # # 对图片进行预处理
    # input_tensor = preprocess(image)
    # input_batch = input_tensor.unsqueeze(0)  # 添加批次维度，因为模型要求输入是一个批次
    #
    # # 将输入数据移到 CPU 或 GPU 上，这取决于你的环境和模型的设定
    # device = torch.device('cpu')  # 或者 torch.device('cuda') 如果你的环境支持 CUDA
    # input_batch = input_batch.to(device)
    #
    # # 将模型设置为评估模式
    #
    # # 使用模型进行推理
    # with torch.no_grad():
    #     output = model(input_batch)
    #
    # # 在此处处理输出，根据模型和任务的不同，输出可以是类别概率、回归值等等
    #
    # # 输出结果示例
    # print(output)
    #
    # # 可以进一步解析输出，例如获取预测的类别或其他信息
    # # 注意：这部分根据你的具体模型输出进行调整
    # # predicted_class = torch.argmax(output, dim=1)
    #
    # # 显示输入图片
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

