MODEL_CONFIG = {
    # 'model_name': 'Resnet101_cbam',
    'model_name': 'MobileViT',
    'num_classes': 2,
    'input_channels': 2,
    'use_cbam': False
}


PARAM_CONFIG = {
    'Resnet18_cbam': {
        'num_classes': 2,
        'input_channels': 2,
        'k_folds': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-2
    },
    'Resnet50_cbam': {
        'num_classes': 2,
        'input_channels': 2,
        'k_folds': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-2
    },
    'Resnet101_cbam': {
        'num_classes': 2,
        'input_channels': 2,
        'k_folds': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-2
    },

    'vit_b_16': {
        'num_classes': 2,
        'input_channels': 2,
        'k_folds': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 0.05
    },

    'DeiT': {
        'num_classes': 2,
        'input_channels': 2,
        'k_folds': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 0.05
    },

    'MobileViT': {
        'num_classes': 2,
        'input_channels': 2,
        'k_folds': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 0.05
    }
}