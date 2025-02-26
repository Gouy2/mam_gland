# from timm.models import create_model

# # 打印完整模型结构
# model = create_model('mobilevit_xs', pretrained=False, num_classes=2)
# # print(model)

# # 打印stem层的具体结构
# print("\nStem layer details:")
# print(model.stem)

from utils.load_data import process_images_for_patients,cache_dataset

def cache_images():
        # 加载所有病人的图像
        # base_path = './data/chaoyang_huigu' 
        base_path = './data/chaoyang_qianzhan_190' 
        patient_images = process_images_for_patients(base_path, target_size=(224, 224), is_mask=False, is_double=False)

        # 缓存预处理后图像
        # cache_dataset(patient_images, f'cache/train_{len(patient_images)}_nonfo.npy', format='npy')  
if __name__ == '__main__':
        
    cache_images()  