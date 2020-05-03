source activate torch
export CUDA_VISIBLE_DEVICES=1
export TRAIN_FILE=/data/bzw/MRC/data/lic2020/dureader_robust-data/dev_words.raw
export TEST_FILE=/data/bzw/MRC/data/lic2020/dureader_robust-data/dev_words.raw

python run_language_modeling.py \
    --output_dir=models \
    --model_type=bert \
    --model_name_or_path=/data/package/chinese_roberta_wwm_large_ext_pytorch \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --block_size 128\
    --gradient_accumulation_steps 8\
    --per_gpu_train_batch_size 4\
    --per_gpu_eval_batch_size 8\
    --learning_rate 1e-5\
    --overwrite_output_dir \
    --doc_stride 64\
    --num_train_epochs 3\
    --mlm_probability 0.10\
    --warmup_proportion 0.1\
    --warmup_steps -1\
    --logging_steps 30\
    --mlm
    
    #--model_name_or_path=./models/checkpoint-206 \
