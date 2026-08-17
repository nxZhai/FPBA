from models.CNNSpot import CNNSpot
from models.DCTA import DCTA
from models.DenseNet import DenseNet
from models.EfficientNet import EfficientNet
from models.MobileNet import MobileNet
from models.Spec import Spec
from models.Swin import Swin_B
from models.ViT import ViT_B_16


def get_model(opt):

    model_name = opt.model
    mode = opt.mode or "eval"

    if model_name == "CNNSpot":
        return CNNSpot(mode)
    if model_name == "DenseNet":
        return DenseNet(mode)
    if model_name == "EfficientNet":
        return EfficientNet(mode)
    if model_name == "MobileNet":
        return MobileNet(mode)
    if model_name == "Spec":
        return Spec(mode)
    if model_name == "DCTA":
        return DCTA(mode, mean=opt.dct_mean, var=opt.dct_var)
    if model_name == "ViT":
        return ViT_B_16(mode)
    if model_name == "Swin":
        return Swin_B(mode)

    raise ValueError(f"model {model_name} not found")
