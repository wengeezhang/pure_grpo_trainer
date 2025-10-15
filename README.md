a pure grpo trainer for huggingface transformers models

# 理念
声明式训练，避免在代码逻辑中嵌入训练细节。

# 训练采样模式
![grpo_sample_img](./img/grpo_sample.png)

# 限制
* only support pytorch
* only support huggingface transformers models
* only support fsdp/fsdp2; deepspeed not supported yet