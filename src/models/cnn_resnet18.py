import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18OCT(nn.Module):

    def __init__(self, num_classes=8, use_pretrained=False):
        super().__init__()

        if use_pretrained:
            w = ResNet18_Weights.DEFAULT
        else:
            w = None
        # luam numa schelete modelului

        backbone = models.resnet18(weights=w)

        in_dim = backbone.fc.in_features
        backbone.fc = nn.Linear(in_dim, num_classes)
        # inlocuim layerul final pt a face calsificare pt cele 8 de boli pe care le avem
        # ne intrezeaza oct urile numa pt cele 8 clase pt clasificare

        self.net = backbone
        # salvam modelul modificat

        mode = "pretrained" if use_pretrained else "from scratch"
        print(f"ResNet18 ready: {num_classes} classes, {mode}")

    def forward(self, x):
        return self.net(x)
        # aici se face procesarea de batch de imagini
        # se trece prin tot modelul si se returneaza outputul final (logits)