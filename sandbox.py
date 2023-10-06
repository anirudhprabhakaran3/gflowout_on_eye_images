from models.resnet_gfn_d import ResNet_GFN
from torchinfo import summary
import torch


class Options:
    def __init__(self):
        self.use_pretrained = True
        self.mlp_dr = 0.3
        self.lr = 0.001
        self.Tune_last_layer_only = True
        self.schedule_milestone = []
        self.BNN = False


opt = Options()

print(opt)

model = ResNet_GFN(opt=opt, num_classes=2)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_d = torch.rand(32, 3, 32, 32).to(device)
y = torch.tensor([1] * 32).to(device)
print(input_d.shape)
print(y.shape)

summary(model, input_data=[input_d, y])
