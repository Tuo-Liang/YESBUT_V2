# Tested on 8x H100 GPUs
# /data/huzhe/workspace/model_card/llava-onevision-qwen2-7b-ov-hf
# llava-hf/llava-v1.6-mistral-7b-hf
# Qwen/Qwen2-VL-7B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --config_file=./accelerate_configs/deepspeed_zero2.yaml \
    sft_qwen2.py  \
    --dataset_name  data/llava-instruct-mix-vsft \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct\
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir save_model/sft-yesbut-moral-mixed-qwem2-vl-7b \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 1536\
    --num_train_epochs 5 \
    --lora_target_modules q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

