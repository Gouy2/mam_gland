from models.resnet import *
from models.vit import *

class ModelFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_model(model_name, **kwargs):
        model_dict = {
            'Resnet18_cbam': Resnet18_cbam,
            'Resnet50_cbam': Resnet50_cbam,
            'Resnet101_cbam': Resnet101_cbam,
            'vit_b_16': vit_b_16,
            'DeiT': DeiT,
            'MobileViT': MobileViT
            
        }
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not found")
        return model_dict[model_name](**kwargs)

