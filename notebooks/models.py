# import torch.nn as nn
# from torchvision import models

# def get_model(name="mobilenet", num_classes=5, pretrained=True):
#     if name == "mobilenet":
#         model = models.mobilenet_v3_large(pretrained=pretrained)
#         model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
#     elif name == "efficientnet":
#         model = models.efficientnet_b4(pretrained=pretrained)
#         model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
#     else:
#         raise ValueError("Model name must be 'mobilenet' or 'efficientnet'")
#     return model

import torch.nn as nn
from torchvision import models

def get_model(model_name='mobilenet'):
    if model_name == 'mobilenet':
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model