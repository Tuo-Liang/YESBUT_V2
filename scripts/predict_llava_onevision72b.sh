data="yesbut_all.json"
image_folder="all_crop"
#--------------------------------------------
write_path_surffix="img.json"
#task options: contradiction | moral_mcq | title_mcq




task="moral_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0,2  python -u  predict_llava_onevision72b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}


task="title_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0,2  python -u  predict_llava_onevision72b.py \
    --read_path ${data} \
    --write_path "results_split/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder}


# # #--------------------------------------------
# write_path_surffix="with_des.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=0,1

# task="moral_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,3  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/llava_onevision72b/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des


# task="title_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,3  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/llava_onevision72b/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des



#--------------------------------------------
# write_path_surffix="_icot.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=0,1

# task="moral_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1   python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


# task="title_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_des\
#     --icot


#--------------------------------------------
# write_path_surffix="con.json"
# #task options: contradiction | moral_mcq | title_mcq
# export CUDA_VISIBLE_DEVICES=0,1

# task="moral_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,3   python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_con


# task="title_mcq"

# echo "==============================="
# echo "llava eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,3  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_con




# prompt_num=0 # Define prompt number dynamically
# #--------------------------------------------
# #task options: contradiction | moral_mcq | title_mcq

# task="moral_mcq"
# write_path_surffix="img.json"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --prompt_num ${prompt_num} 

# task="title_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --prompt_num ${prompt_num} 

# #--------------------------------------------
# write_path_surffix="with_des.json"

# task="moral_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_des \
#     --prompt_num ${prompt_num} 

# task="title_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_des \
#     --prompt_num ${prompt_num} 

# #--------------------------------------------
# write_path_surffix="con.json"

# task="moral_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_con \
#     --prompt_num ${prompt_num} 

# task="title_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_con \
#     --prompt_num ${prompt_num} 

# wait

# prompt_num=1 # Define prompt number dynamically
# #--------------------------------------------
# #task options: contradiction | moral_mcq | title_mcq

# task="moral_mcq"
# write_path_surffix="img.json"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --prompt_num ${prompt_num} 

# task="title_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --prompt_num ${prompt_num} 

# #--------------------------------------------
# write_path_surffix="with_des.json"

# task="moral_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_des\
#     --prompt_num ${prompt_num} 

# task="title_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_des \
#     --prompt_num ${prompt_num} 

# #--------------------------------------------
# write_path_surffix="con.json"

# task="moral_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_con \
#     --prompt_num ${prompt_num} 

# task="title_mcq"

# echo "==============================="
# echo " eval"
# echo "==============================="
# CUDA_VISIBLE_DEVICES=0,1  python -u  predict_llava_onevision72b.py \
#     --read_path ${data} \
#     --write_path "results_split/results_split_llava_onevision72b_${task}_prompt_num_${prompt_num}_${write_path_surffix}" \
#     --task ${task} \
#     --image_folder ${image_folder} \
#     --gen_con \
#     --prompt_num ${prompt_num} 

