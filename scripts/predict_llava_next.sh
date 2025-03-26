data="yesbut_all.json"
image_folder="all_crop"
# #--------------------------------------------
# write_path_surffix="img.json"
# #task options: contradiction | moral_mcq | title_mcq




# task="moral_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=1 python -u  predict_llava_next.py \
#     --read_path ${data} \
#     --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}


# task="title_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=1 python -u  predict_llava_next.py \
#     --read_path ${data} \
#     --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}


# #--------------------------------------------
write_path_surffix="with_des.json"
#task options: contradiction | moral_mcq | title_mcq


task="moral_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=1 python -u  predict_llava_next.py \
    --read_path ${data} \
    --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_des&


task="title_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=1 python -u  predict_llava_next.py \
    --read_path ${data} \
    --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_des&
wait



#--------------------------------------------
# write_path_surffix="_icot.json"
# #task options: contradiction | moral_mcq | title_mcq
# 

# task="moral_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# python -u  predict_llava_next.py \
#     --read_path ${data} \
#     --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


# task="title_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# python -u  predict_llava_next.py \
#     --read_path ${data} \
#     --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


#--------------------------------------------
write_path_surffix="_con.json"
#task options: contradiction | moral_mcq | title_mcq


task="moral_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=1 python -u  predict_llava_next.py \
    --read_path ${data} \
    --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_con&


task="title_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=1 python -u  predict_llava_next.py \
    --read_path ${data} \
    --write_path "results/results_llava_next_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}\
    --gen_con&

wait