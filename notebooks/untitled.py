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
