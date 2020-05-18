export DATA_DIR=/home/shikib/alexa-prize-topical-chat-dataset/lm_finetune/pc_data/both/

CUDA_VISIBLE_DEVICES=0 python3 train_understandable.py \
    --per_gpu_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --per_gpu_eval_batch_size 16 \
    --save_steps 10000  \
    --num_train_epochs 3 \
    --output_dir=pc_both \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --data_dir $DATA_DIR \
    --do_train \
    --do_eval \
    --task_name=qqp \
