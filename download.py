import os
import argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


parser = argparse.ArgumentParser(description='dowld model from huggingface')  # 创建解析器
parser.add_argument('--model_name', type=str, required=True, help='model name, e.g. stabilityai/sdxl-turbo')  # 添加参数
parser.add_argument('--save_path', type=str, help='save_path')  # 添加参数
args = parser.parse_args()  # 解析参数
# print(args)


model_name = args.model_name

if args.save_path is None:
    save_path = model_name.split("/")[-1]
else:
    save_path = args.save_path

commd = f'huggingface-cli download --local-dir-use-symlinks False {model_name} --local-dir {save_path}'
os.system(commd)

#inspect

url = f'https://huggingface.co/{model_name}/tree/main'