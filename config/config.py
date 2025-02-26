MODEL_CONFIG = {
    'model_name': 'Resnet18_cbam',
    # 'model_name': 'Densenet121_cbam',
    'num_classes': 4,
    'input_channels': 2,
    'k_folds': 5,
    'batch_size': 16,
    'num_epochs': 50,
    'use_cbam': True
}


PARAM_CONFIG = {
    'Resnet18_cbam': {
        'num_classes': MODEL_CONFIG['num_classes'],
        'input_channels': MODEL_CONFIG['input_channels'],
        'k_folds': MODEL_CONFIG['k_folds'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'num_epochs': MODEL_CONFIG['num_epochs'],
        'lr': 1e-3,
        'weight_decay': 1e-2
    },
    'Resnet50_cbam': {
        'num_classes': MODEL_CONFIG['num_classes'],
        'input_channels': MODEL_CONFIG['input_channels'],
        'k_folds': MODEL_CONFIG['k_folds'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'num_epochs': MODEL_CONFIG['num_epochs'],
        'lr': 1e-3,
        'weight_decay': 1e-2
    },
    'Resnet101_cbam': {
        'num_classes': MODEL_CONFIG['num_classes'],
        'input_channels': MODEL_CONFIG['input_channels'],
        'k_folds': MODEL_CONFIG['k_folds'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'num_epochs': MODEL_CONFIG['num_epochs'],
        'lr': 1e-3,
        'weight_decay': 1e-2
    },

    'vit_b_16': {
        'num_classes': MODEL_CONFIG['num_classes'],
        'input_channels': MODEL_CONFIG['input_channels'],
        'k_folds': MODEL_CONFIG['k_folds'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'num_epochs': MODEL_CONFIG['num_epochs'],
        'lr': 1e-4,
        'weight_decay': 0.05
    },

    'DeiT': {
        'num_classes': MODEL_CONFIG['num_classes'],
        'input_channels': MODEL_CONFIG['input_channels'],
        'k_folds': MODEL_CONFIG['k_folds'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'num_epochs': MODEL_CONFIG['num_epochs'],
        'lr': 1e-4,
        'weight_decay': 0.05
    },

    'MobileViT': {
        'num_classes': MODEL_CONFIG['num_classes'],
        'input_channels': MODEL_CONFIG['input_channels'],
        'k_folds': MODEL_CONFIG['k_folds'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'num_epochs': MODEL_CONFIG['num_epochs'],
        'lr': 1e-4,
        'weight_decay': 0.05
    },

    'Densenet121_cbam': {
        'num_classes': MODEL_CONFIG['num_classes'],
        'input_channels': MODEL_CONFIG['input_channels'],
        'k_folds': MODEL_CONFIG['k_folds'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'num_epochs': MODEL_CONFIG['num_epochs'],
        'lr': 1e-4,
        'weight_decay': 1e-2
    }
}