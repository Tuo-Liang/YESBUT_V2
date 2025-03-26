data="yesbut_all.json"
image_folder="all_crop"

# write_path_surffix="gt_des.json"
# #task options: contradiction | moral_mcq | title_mcq


# task="moral_mcq"

# echo "==============================="
# echo "deepseek_r1 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0  python -u  predict_deepseek_r1.py \
#     --read_path ${data} \
#     --write_path "results/deepseek_r1/results_deepseek_r1_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gt_des &


# task="title_mcq"

# echo "==============================="
# echo "deepseek_r1 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=1 python -u  predict_deepseek_r1.py \
#     --read_path ${data} \
#     --write_path "results/deepseek_r1/results_deepseek_r1_"${task}"_"${write_path_surffix} \
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
# echo "deepseek_r1 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0  python -u  predict_deepseek_r1.py \
#     --read_path ${data} \
#     --write_path "results/deepseek_r1/results_deepseek_r1"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gt_des \
#     --gen_con &


# task="title_mcq"

# echo "==============================="
# echo "deepseek_r1 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=1 python -u  predict_deepseek_r1.py \
#     --read_path ${data} \
#     --write_path "results/deepseek_r1/results_deepseek_r1"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gt_des \
#     --gen_con &
# wait

write_path_surffix="with_social_info.json"
#task options: contradiction | moral_mcq | title_mcq


task="moral_mcq"

echo "==============================="
echo "deepseek_r1 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_deepseek_r1.py \
    --read_path ${data} \
    --write_path "results/results_deepseek_r1_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
    --with_social_info 


task="title_mcq"

echo "==============================="
echo "deepseek_r1 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0 python -u  predict_deepseek_r1.py \
    --read_path ${data} \
    --write_path "results/results_deepseek_r1_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
    --with_social_info 
