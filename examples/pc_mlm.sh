export EVAL_FILE=undr/pc_test.lm

CUDA_VISIBLE_DEVICES=1 python3 run_lm_finetuning.py \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --output_dir=roberta_pc3 \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --train_data_file=$EVAL_FILE \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --mlm
