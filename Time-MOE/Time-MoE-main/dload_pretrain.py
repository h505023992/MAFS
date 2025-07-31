import torch
from transformers import AutoModelForCausalLM

# 模拟输入
context_length = 12
batch_size = 2
normed_seqs = torch.randn(batch_size, context_length).to(torch.float16).to("cuda:1")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cuda:1",
    attn_implementation='flash_attention_2',
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 打印结构
print(model)

# 生成预测
prediction_length = 6
output = model.generate(
    inputs_embeds=normed_seqs,
    max_new_tokens=prediction_length
)
normed_predictions = output[:, -prediction_length:]
