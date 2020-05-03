# Masked_LM,
requirements:
dataclasses=0.7.dist
transformers=2.8.0

transformers需要升级到最新版本，我试了新版本的transformers在前面几个项目上也能用。所以直接更新已有环境中的transformers即可，不需要再创建一个新的环境。

可能需要设置的训练参数
output_dir='models'
overwrite_output_dir=True
do_train=True
do_eval=True
do_predict=False
per_gpu_train_batch_size=4
per_gpu_eval_batch_size=8
gradient_accumulation_steps=8
learning_rate=1e-05
weight_decay=0.0
adam_epsilon=1e-08
max_grad_norm=1.0
num_train_epochs=3.0
warmup_steps=-1
logging_dir=None
seed=42
warmup_proportion=0.1
