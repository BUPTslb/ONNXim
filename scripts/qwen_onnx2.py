from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_folder="../models/qwen/"
onnx_model_path = os.path.join(output_folder, "qwen2.5-1.5b.onnx")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import torch

# 准备一个虚拟输入张量，假设输入长度为128
dummy_input = torch.randint(0, 100, (1, 128), dtype=torch.int).to(model.device)

torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    input_names=["input_ids"],
    output_names=["output"],
    opset_version=14  # 选择合适的 opset 版本
)

import onnx
from onnxruntime.transformers import optimizer

# 加载 ONNX 模型
onnx_model = onnx.load(onnx_model_path)

# 打印模型结构
# print(onnx.helper.printable_graph(onnx_model.graph))

# 检查模型是否有效
onnx.checker.check_model(onnx_model_path)

print("模型验证成功！")



