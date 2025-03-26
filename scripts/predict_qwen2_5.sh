data="yesbut_all.json"
image_folder="all_crop"

# write_path_surffix="gt_des.json"
# #task options: contradiction | moral_mcq | title_mcq


# task="moral_mcq"

# echo "==============================="
# echo "qwen2_5 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=2  python -u  predict_qwen2_5.py \
#     --read_path ${data} \
#     --write_path "results/qwen2_5_72b/results_qwen2_5_72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gt_des &


# task="title_mcq"

# echo "==============================="
# echo "qwen2_5 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=3 python -u  predict_qwen2_5.py \
#     --read_path ${data} \
#     --write_path "results/qwen2_5_72b/results_qwen2_5_72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gt_des &
# wait


# #--------------------------------------------
# write_path_surffix="con.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=3

# task="moral_mcq"

# echo "==============================="
# echo "qwen2_5 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=2  python -u  predict_qwen2_5.py \
#     --read_path ${data} \
#     --write_path "results/qwen2_5_72b/results_qwen2_5_72b"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gt_des \
#     --gen_con &


# task="title_mcq"

# echo "==============================="
# echo "qwen2_5 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=3 python -u  predict_qwen2_5.py \
#     --read_path ${data} \
#     --write_path "results/qwen2_5_72b/results_qwen2_5_72b"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gt_des \
#     --gen_con &
# wait



write_path_surffix="with_social_info.json"
#task options: contradiction | moral_mcq | title_mcq


task="moral_mcq"

echo "==============================="
echo "qwen2_5 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2  python -u  predict_qwen2_5.py \
    --read_path ${data} \
    --write_path "results/qwen2_5_72b/results_qwen2_5_72b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
    --with_social_info 


task="title_mcq"

echo "==============================="
echo "qwen2_5 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2 python -u  predict_qwen2_5.py \
    --read_path ${data} \
    --write_path "results/qwen2_5_72b/results_qwen2_5_72b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
    --with_social_info