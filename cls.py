import os
import glob

# 获取当前文件夹中的所有.txt文件
# files = glob.glob("./image/*.png")
# files = glob.glob("*.pth")

files = glob.glob("./model_save/*.pth")

for file in files:
    os.remove(file)
    print(f"{file} 已删除")