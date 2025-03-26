data="yesbut_all.json"
image_folder="all_crop"

write_path_surffix="gen_des.json"

model="llava_next13b"
task="moral_mcq"

echo "==============================="
echo "gpt4 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_gpt4.py \
    --read_path ${data} \
    --write_path "results/gpt4/results_gpt4_"${model}"_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
    --using_gen_des \
    --gen_des_file "generated_text/llava_next13b_des.json" &

task="title_mcq"

echo "==============================="
echo "gpt4 eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=1 python -u  predict_gpt4.py \
    --read_path ${data} \
    --write_path "results/gpt4/results_gpt4_"${model}"_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
    --using_gen_des \
    --gen_des_file "generated_text/llava_next13b_des.json" &




write_path_surffix="con.json"

model="llava_next13b"
task="moral_mcq"

echo "==============================="
echo "gpt4 eval"
echo "==============================="
python -u  predict_gpt4.py \
    --read_path ${data} \
    --write_path "results/gpt4/results_gpt4_"${model}"_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
 --using_gen_des \
     --gen_des_file "generated_text/llava_next13b_des.json" \
    --gen_con &

task="title_mcq"

echo "==============================="
echo "gpt4 eval"
echo "==============================="
python -u  predict_gpt4.py \
    --read_path ${data} \
    --write_path "results/gpt4/results_gpt4_"${model}"_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --image_folder ${image_folder} \
 --using_gen_des \
     --gen_des_file "generated_text/llava_next13b_des.json" \
    --gen_con &

wait




# #--------------------------------------------
# write_path_surffix="description.json"
# #task options: contradiction | moral_mcq | title_mcq


# task="moral_mcq"

# echo "==============================="
# echo "gpto1 eval"
# echo "==============================="
#  python -u  predict_gpto1.py \
#     --read_path ${data} \
#     --write_path "results/results_gpto1_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} &


# task="title_mcq"

# echo "==============================="
# echo "gpto1 eval"
# echo "==============================="
# python -u  predict_gpto1.py \
#     --read_path ${data} \
#     --write_path "results/results_gpto1_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder} &




# #--------------------------------------------
# write_path_surffix="con.json"
# # task options: contradiction | moral_mcq | title_mcq

# task="moral_mcq"

# echo "==============================="
# echo "gpto1 eval"
# echo "==============================="
# python -u  predict_gpto1.py \
#     --read_path ${data} \
#     --write_path "results/results_gpto1_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_con &


# task="title_mcq"

# echo "==============================="
# echo "gpto1 eval"
# echo "==============================="
# python -u  predict_gpto1.py \
#     --read_path ${data} \
#     --write_path "results/results_gpto1_"${task}"_"${write_path_surffix} \
#     --task ${task} \
#     --image_folder ${image_folder}\
#     --gen_con &

# wait