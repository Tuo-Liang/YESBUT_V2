data="yesbut_all.json"
image_folder="all_crop"
# #--------------------------------------------
# write_path_surffix="img.json"
# #task options: contradiction | moral_mcq | title_mcq


# task="moral_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0  python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} &


# task="title_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0 python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} &

# wait

# #--------------------------------------------
# write_path_surffix="with_des.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=3

# task="moral_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0   python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des &


# task="title_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0 python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des &
# wait

# #--------------------------------------------
# # write_path_surffix="_icot.json"
# # #task options: contradiction | moral_mcq | title_mcq
# # export CUDA_VISIBLE_DEVICES=3

# # task="moral_mcq"

# # echo "==============================="
# # echo "qwen2 eval"
# # echo "==============================="
# # CUDA_VISIBLE_DEVICES=3  python -u  predict_qwen2_7b.py \
# #     --read_path ${data} \
# #     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
# #     --task ${task} \
# #     --image_folder ${image_folder}\
# #     --gen_des\
# #     --icot


# # task="title_mcq"

# # echo "==============================="
# # echo "qwen2 eval"
# # echo "==============================="
# # CUDA_VISIBLE_DEVICES=3 python -u  predict_qwen2_7b.py \
# #     --read_path ${data} \
# #     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
# #     --task ${task} \
# #     --image_folder ${image_folder}\
# #     --gen_des\
# #     --icot


# #--------------------------------------------
# write_path_surffix="con.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=3

# task="moral_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0  python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_con &


# task="title_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0 python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_con &

# wait


# write_path_surffix="gt_des.json"
# #task options: contradiction | moral_mcq | title_mcq

# task="moral_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=1  python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results/qwen2_7b/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gt_des &


# task="title_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=1 python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results/qwen2_7b/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gt_des &

# wait



write_path_surffix="with_social_info.json"
#task options: contradiction | moral_mcq | title_mcq

# task="moral_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=1  python -u  predict_qwen2_7b.py \
#     --read_path ${data} \
#     --write_path "results/qwen2_7b/results_qwen_2_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --with_social_info &


task="title_mcq"

echo "==============================="
echo "qwen2 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=1 python -u  predict_qwen2_7b.py \
    --read_path ${data} \
    --write_path "results_nosplit/qwen2_7b/results_qwen_2_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
    --with_social_info 

