from timm.models import create_model

# 打印完整模型结构
model = create_model('mobilevit_xs', pretrained=False, num_classes=2)
# print(model)

# 打印stem层的具体结构
print("\nStem layer details:")
print(model.stem)
