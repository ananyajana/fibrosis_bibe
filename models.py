import torch
from torch import nn
from torchvision import models as torch_models
import torch.nn.functional as F


class BaselineNet(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(BaselineNet, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.resnet = ResNet_extractor(layers, train_res4)
        self.classifier = Classifier(dim_dict[layers], num_classes)

    def forward(self, x):
        x = self.resnet(x)
        out = self.classifier(x)
        return out


class MidFusion(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(MidFusion, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.resnet_ct = ResNet_extractor(layers, train_res4)
        self.resnet_patho = ResNet_extractor(layers, train_res4)
        self.classifier_ct = Classifier(dim_dict[layers], num_classes)
        self.classifier_patho = Classifier(dim_dict[layers], num_classes)
        self.classifier_joint = Classifier(dim_dict[layers], num_classes)

    def forward(self, x_ct, x_patho, loss_type='single'):
        x_ct = self.resnet_ct(x_ct)
        x_patho = self.resnet_patho(x_patho)
        x_cat = torch.cat([x_ct, x_patho], dim=0)
        out_joint = self.classifier_joint(x_cat)
        if loss_type == 'single':
            return out_joint
        else:
            out_ct = self.classifier_ct(x_ct)
            out_patho = self.classifier_patho(x_patho)
            out_joint = self.classifier_joint(x_cat)
            return out_ct, out_patho, out_joint


class LateFusion(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(LateFusion, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}
        self.resnet_ct = ResNet_extractor(layers, train_res4)
        self.resnet_patho = ResNet_extractor(layers, train_res4)
        self.classifier_ct = Classifier(dim_dict[layers], num_classes)
        self.classifier_patho = Classifier(dim_dict[layers], num_classes)
        self.classifier_joint = nn.Linear(256, num_classes)

    def forward(self, x_ct, x_patho, loss_type='single'):
        x_ct = self.resnet_ct(x_ct)
        x_patho = self.resnet_patho(x_patho)
        out_ct, feat_ct = self.classifier_ct(x_ct, feat=True)
        out_patho, feat_patho = self.classifier_patho(x_patho, feat=True)
        x_cat = torch.cat([feat_ct, feat_patho], dim=1)
        out_joint = self.classifier_joint(x_cat)
        if loss_type == 'single':
            return out_joint
        else:
            return out_ct, out_patho, out_joint


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, feat=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x1, _ = torch.max(x, dim=0)  # max pooling
        x2 = x.mean(dim=0, keepdim=True)
        x = self.fc3(x2)

        if feat:
            return x, x2
        else:
            return x


class ResNet_extractor(nn.Module):
    def __init__(self, layers=50, train_res4=True):
        super().__init__()
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

        for param in self.resnet.parameters():
            param.requires_grad = False

        if train_res4:  # train res4
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.avgpool.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
