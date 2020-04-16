export DATA_DIR=../../../labels/both/

CUDA_VISIBLE_DEVICES=1 python3 train_understandable.py \
    --per_gpu_eval_batch_size=1 \
    --output_dir=ctx \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --data_dir $DATA_DIR \
    --do_eval \
    --task_name=qqp \
