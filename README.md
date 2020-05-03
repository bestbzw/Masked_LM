# Masked_LM,

##requirements:

dataclasses=0.7.dist

transformers=2.8.0

##transformers需要升级到最新版本，我试了新版本的transformers在前面几个项目上也能用。所以直接更新已有环境中的transformers即可，不需要再创建一个新的环境。

##可能需要设置的训练参数

output_dir

overwrite_output_dir

do_train

do_eval

do_predict

per_gpu_train_batch_size

per_gpu_eval_batch_size

gradient_accumulation_steps

learning_rate

weight_decay

adam_epsilon

max_grad_norm

num_train_epochs

warmup_steps

logging_dir

seed

warmup_proportion
