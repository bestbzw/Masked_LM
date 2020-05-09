source activate torch
export CUDA_VISIBLE_DEVICES=0
export TRAIN_FILE=/data/bzw/MRC/data/lic2020/extend_data/lic_dev_test1.raw
export TEST_FILE=/data/bzw/MRC/data/lic2020/dureader_robust-data/dev_words.raw

python run_language_modeling.py \
    --output_dir=baidu+atec_zybang1and2_3longer+dev_test1_models \
    --model_type=bert \
    --model_name_or_path=./baidu+atec_zybang1and2_3longer_models/baidu+atec_zybang_epoch2/ \
    --logging_dir=baidu+atec_zybang1and2_3longer+dev_test1_models \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --block_size 256\
    --gradient_accumulation_steps 32\
    --per_gpu_train_batch_size 2\
    --per_gpu_eval_batch_size 8\
    --learning_rate 1e-5\
    --overwrite_output_dir \
    --doc_stride 128\
    --num_train_epochs 2\
    --mlm_probability 0.10\
    --warmup_proportion 0.1\
    --warmup_steps -1\
    --mlm
    
    #--model_name_or_path=./models/checkpoint-206 \
