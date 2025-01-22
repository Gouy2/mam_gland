from models.resnet import *

class ModelFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_model(model_name, **kwargs):
        model_dict = {
            'Resnet18_cbam': Resnet18_cbam,
            'Resnet50_cbam': Resnet50_cbam,
            'Resnet101_cbam': Resnet101_cbam,
            
        }
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not found")
        return model_dict[model_name](**kwargs)

