data="yesbut_all.json"
image_folder="all_crop"
#--------------------------------------------
write_path_surffix="img.json"
#task options: contradiction | moral_mcq | title_mcq


export CUDA_VISIBLE_DEVICES=2

task="moral_mcq"

echo "==============================="
echo "qwen2 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2  python -u  predict_qwen2_72b.py \
    --read_path ${data} \
    --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} 


task="title_mcq"

echo "==============================="
echo "qwen2 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2 python -u  predict_qwen2_72b.py \
    --read_path ${data} \
    --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} 


# #--------------------------------------------
write_path_surffix="with_des.json"
#task options: contradiction | moral_mcq | title_mcq
export CUDA_VISIBLE_DEVICES=2

task="moral_mcq"

echo "==============================="
echo "qwen2 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2 python -u  predict_qwen2_72b.py \
    --read_path ${data} \
    --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_des 


task="title_mcq"

echo "==============================="
echo "qwen2 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2 python -u  predict_qwen2_72b.py \
    --read_path ${data} \
    --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_des 

#--------------------------------------------
# write_path_surffix="_icot.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=3

# task="moral_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=3  python -u  predict_qwen2_72b.py \
#     --read_path ${data} \
#     --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


# task="title_mcq"

# echo "==============================="
# echo "qwen2 eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=3 python -u  predict_qwen2_72b.py \
#     --read_path ${data} \
#     --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


#--------------------------------------------
write_path_surffix="con.json"
#task options: contradiction | moral_mcq | title_mcq
export CUDA_VISIBLE_DEVICES=2

task="moral_mcq"

echo "==============================="
echo "qwen2 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2  python -u  predict_qwen2_72b.py \
    --read_path ${data} \
    --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_con 


task="title_mcq"

echo "==============================="
echo "qwen2 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=2 python -u  predict_qwen2_72b.py \
    --read_path ${data} \
    --write_path "results/results_qwen2_72b"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_con 

