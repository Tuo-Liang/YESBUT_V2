data="yesbut_all.json"
image_folder="all_crop"
#--------------------------------------------
write_path_surffix="img.json"
#task options: contradiction | moral_mcq | title_mcq


export CUDA_VISIBLE_DEVICES=0

task="moral_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0   python -u  predict_llava_onevision0.5b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}&


task="title_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_llava_onevision0.5b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}&


# #--------------------------------------------
write_path_surffix="with_des.json"
#task options: contradiction | moral_mcq | title_mcq
export CUDA_VISIBLE_DEVICES=0

task="moral_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0   python -u  predict_llava_onevision0.5b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_des&


task="title_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_llava_onevision0.5b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_des&



#--------------------------------------------
# write_path_surffix="_icot.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=0

# task="moral_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0   python -u  predict_llava_onevision0.5b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


# task="title_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0  python -u  predict_llava_onevision0.5b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


#--------------------------------------------
write_path_surffix="con.json"
#task options: contradiction | moral_mcq | title_mcq
export CUDA_VISIBLE_DEVICES=0

task="moral_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0   python -u  predict_llava_onevision0.5b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_con &


task="title_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_llava_onevision0.5b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision0.5b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_con&

wait