from models.CNNSpot import CNNSpot
from models.DenseNet import DenseNet
from models.Swin import Swin_B


def get_model(opt):
    
    model_name = opt.model


    if model_name == "CNNSpot":
        model = CNNSpot()
    elif model_name == "DenseNet":
        model = DenseNet()
    elif model_name == "Swin":
        model = Swin_B()
    else:
        raise ValueError(f"model {model_name} not found")

    return model