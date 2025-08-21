from models.CNNSpot import CNNSpot
from models.DenseNet import DenseNet
from models.Swin import Swin_B


def get_model(opt):
    
    model_name = opt.model
    mode = opt.mode

    if model_name == "CNNSpot":
        model = CNNSpot(mode)
    elif model_name == "DenseNet":
        model = DenseNet(mode)
    elif model_name == "Swin":
        model = Swin_B(mode)
    else:
        raise ValueError(f"model {model_name} not found")

    return model