import numpy as np
import torch
import timm

filepath = "/Users/isaiah/datasets/kaggle_mammograph/taiseng/results/20230225/densenet_best_02.pth"
model_params = {
    "in_chans": 1,
    "num_classes": 4,
    "pretrained": False,
    "drop_rate": 0.5,
}
model = timm.create_model("resnet18", **model_params)
model.load_state_dict(torch.load(filepath, map_location=torch.device("mps"))["model"])
no_params = np.sum([ np.prod(list(i.shape)) for i in model.parameters() ])
print(f"ResNet18 parameter count: {no_params}")
