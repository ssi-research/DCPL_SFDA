import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import swin_transformer as timm_swin
from torchvision import models
from torchvision.models import swin_transformer



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# @register_model
# def swin_base_patch4_window7_224(pretrained=False, **kwargs) -> SwinTransformer:
#     """ Swin-B @ 224x224
#     """
#     model_args = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
#     return _create_swin_transformer(
#         'swin_base_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


class SwinBase(nn.Module):
    def __init__(self, in_22k_1k=False, in_timm_1k=False):
        super(SwinBase, self).__init__()
        self.in_22k_1k = in_22k_1k
        self.in_timm_1k = in_timm_1k
        if in_22k_1k:
            self.model_swin = timm_swin.swin_base_patch4_window7_224(pretrained=True)
        elif in_timm_1k:
            self.model_swin = timm_swin.swin_base_patch4_window7_224_1K(pretrained=True)
        else:
            self.model_swin = swin_transformer.swin_b(weights=swin_transformer.Swin_B_Weights.IMAGENET1K_V1).cuda()
            self.model_swin.head = torch.nn.Identity()

        # self.in_features = 1024

    def forward(self, x):
        if self.in_22k_1k or self.in_timm_1k:
            x = self.model_swin.forward_features(x)
        else:
            x = self.model_swin(x)
        x = x.view(x.size(0), -1)
        return x


class Identity(nn.Module):

    def __init__(self, sub=1.0):
        super(Identity, self).__init__()
        self.sub = sub

    def forward(self, x):
        return x * self.sub


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn" or self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.bn(x)
        if self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.relu(x)
        if self.type == "bn_relu_drop":
            x = self.dropout(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear", add_DCPL=False, num_annotators=1):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = torch.nn.utils.weight_norm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)

            x = torch.nn.functional.normalize(x, dim=1, p=2)
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)

        return x


class conf_mat(nn.Module):
    def __init__(self, class_num, num_annotators=1, normalize=True, is_data_dependent=False, bottleneck_dim=256,
                 applySoftmax=False,
                 use_sigmoid=False, beta=2):
        super(conf_mat, self).__init__()

        self.beta = beta
        self.applySoftmax = applySoftmax
        self.normalize = normalize
        self.use_sigmoid = use_sigmoid
        # w_init = torch.tensor(np.stack([6.0 * np.eye(class_num) - 5.0 for j in range(num_annotators)]), dtype=torch.float32)
        self.CM = nn.Parameter(torch.zeros((num_annotators, class_num, class_num)))
        for j in range(num_annotators):
            nn.init.eye_(self.CM[j, :, :])

        cm_sigmoid_data = -6.0 * torch.ones(self.CM[0, :, :].shape)
        I = torch.eye(len(cm_sigmoid_data))
        cm_sigmoid_data = (1 - I) * cm_sigmoid_data + I * 3
        self.cm_sigmoid = nn.Parameter(cm_sigmoid_data)

        self.relu = nn.ReLU()

        self.class_num = class_num
        self.fc = nn.Linear(bottleneck_dim, class_num * class_num)
        self.fc.apply(init_weights)
        self.is_data_dependent = is_data_dependent

        self.cm_data_reinitialized = False

    def reinit_cm_matrix_data(self, cm_data):
        if self.cm_data_reinitialized:
            raise Exception('cm data already initialized')
        if self.use_sigmoid:
            Z_sigmoid_inv = torch.log(cm_data / (1 - cm_data + 1e-9))
            self.cm_sigmoid.data = Z_sigmoid_inv.reshape(self.cm_sigmoid.data.shape)
            self.cm_data_reinitialized = True
        else:
            raise Exception('reinitializing data matrix not supported for formulation other than sigmoid.')

    def forward(self, x):
        if self.is_data_dependent:
            x = self.fc(x)
            x = torch.reshape(x, (x.size()[0], self.class_num, self.class_num))
            x = F.softmax(x, dim=2)
        else:
            # ensure positivity
            if self.applySoftmax:
                x = F.softmax(self.CM, dim=2)
            elif self.use_sigmoid:
                x = torch.nn.functional.sigmoid(self.cm_sigmoid).reshape(self.CM.shape)
            else:
                x = F.softplus(self.CM, beta=self.beta)
            # ensure each row sums to one
            if self.normalize:
                x = torch.nn.functional.normalize(x, dim=2, p=1)

        return x


class modnetSHOT(nn.Module):

    def __init__(self, num_C, cnn, E_dims, apply_wn=False, type_bottleneck="bn", add_DCPL=False, num_annotators=1,
                 normalize=True, is_data_dependent=False, applySoftmax=False, beta=2, use_sigmoid=False):

        super(modnetSHOT, self).__init__()

        # Frozen initial conv layers
        if cnn == 'resnet50' or cnn == 'resnet101':
            self.M = ResBase(res_name=cnn)
            feature_dim = 2048
        elif cnn == 'SwinBase1K':
            self.M = SwinBase()
            feature_dim = 1024
        elif cnn == 'SwinBase22':
            self.M = SwinBase(in_22k_1k=True)
            feature_dim = 1024
        elif cnn == 'SwinBase':
            self.M = SwinBase(in_22k_1k=False, in_timm_1k=True)
            feature_dim = 1024
        else:
            raise NotImplementedError('Not implemented for ' + str(cnn))

        self.E = feat_bootleneck(feature_dim=feature_dim, bottleneck_dim=E_dims, type=type_bottleneck)

        if apply_wn:
            self.G = feat_classifier(num_C, E_dims, type="wn", add_DCPL=add_DCPL)
        else:
            self.G = feat_classifier(num_C, E_dims, type="linear", add_DCPL=add_DCPL)

        self.CM = conf_mat(class_num=num_C, num_annotators=num_annotators, normalize=normalize,
                           is_data_dependent=is_data_dependent, bottleneck_dim=E_dims,
                           applySoftmax=applySoftmax, beta=beta, use_sigmoid=use_sigmoid)

        self.components = {
            'M': self.M,
            'E': self.E,
            'G': self.G,
            'CM': self.CM,
        }

    def forward(self, x, which_fext='original'):
        raise NotImplementedError('Implemented a custom forward in train loop')
