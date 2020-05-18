export TRAIN_FILE=/home/shikib/convai/lm/train.lm
export TEST_FILE=/home/shikib/convai/lm/test.lm

CUDA_VISIBLE_DEVICES=1 python3 run_lm_finetuning.py \
    --per_gpu_train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --per_gpu_eval_batch_size=1 \
    --save_steps=10000 \
    --num_train_epochs=3 \
    --output_dir=roberta_pc_3epochs \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --train_data_file=$TRAIN_FILE \
    --do_train \
    --eval_data_file=$TEST_FILE \
    --mlm
