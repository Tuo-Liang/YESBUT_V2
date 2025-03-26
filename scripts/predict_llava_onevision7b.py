import requests
from PIL import Image
import json
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig


device = 'cuda'


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="int8",
    bnb_8bit_compute_dtype=torch.float16,
)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # 改为加载 4 位量化
#     bnb_4bit_quant_type="nf4",  # 使用 int4 量化类型（常用 "nf4"）
#     bnb_4bit_compute_dtype=torch.float16,  # 计算时使用 float16
# )


model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
#model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
#model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
print(f"Using model: {model_id}")

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    use_flash_attention_2=True,
    # quantization_config=quantization_config, 
    device_map ='auto'
)

processor = AutoProcessor.from_pretrained(model_id)

def resize_image_to_512(image_path):
    """
    将图片的宽或高调整到 512 像素以内，同时保持宽高比。
    
    参数:
        image_path (str): 输入图片的路径。
    
    返回:
        resized_img (PIL.Image.Image): 调整大小后的图片对象。
    """
    # 打开图片
    with Image.open(image_path) as img:
        # 获取原始宽高
        original_width, original_height = img.size
        
        # 确定缩放比例
        if max(original_width, original_height) > 512:
            scaling_factor = 512 / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
        else:
            # 如果宽高都小于512，保持原始大小
            new_width, new_height = original_width, original_height
        
        # 调整大小
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return resized_img
    


def get_prompt(instruction,image_num=2):
    if image_num==1:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--gen_des', action= "store_true")
    parser.add_argument('--gen_con', action= "store_true")
    parser.add_argument('--icot', action= "store_true")
    args = parser.parse_args()
    print(args)

    data = json.load(open(args.read_path))
    results = []
    

    for sample in data:
        image_file = sample["image_file"]
        image_path1= "cropped_images/panel/"+image_file.split(".")[0]+"_panel_1.png"
        image_path2= "cropped_images/panel/"+image_file.split(".")[0]+"_panel_2.png"
        img1 = resize_image_to_512(image_path1)
        img2 = resize_image_to_512(image_path2)
        img = [img1,img2]
        # image_path = args.image_folder + "/" + image_file
        # img = Image.open(image_path)

        cur_caption =None
        contradiction = None
        instruction = None  

        if args.gen_des:
            print("---------generate caption--------")
            #get prompt
            #prompt=get_prompt("Literally describe these two panels respectively.")
            prompt=get_prompt("Describe this image specifically.")
            
            print("[Prompt]:",prompt)
            inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)
            inputs = inputs.to(torch.float16)

            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result=processor.decode(output[0][2:], skip_special_tokens=True)

            if "assistant" in result:
                cur_caption = result.split('assistant')[1].strip()
            print('[Gen_caption]:',cur_caption)
            sample['gen_des']=cur_caption


        if args.gen_con and args.icot!=True:
            print("---------generate contradiction--------")
            prompt=get_prompt("Tell me the contradiction between the two panels in 2 sentences.")

            print("[Prompt]:",prompt)
            inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)
            inputs = inputs.to(torch.float16)

            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result=processor.decode(output[0][2:], skip_special_tokens=True)

            if "assistant" in result:
                contradiction = result.split('assistant')[1].strip()

            print('[Gen_contradiction]:',contradiction)
            sample['gen_con']=contradiction

            
        #get question
        question =sample[args.task]

        #only img input
        if args.gen_des!=True and args.gen_con!=True:
            instruction = f"Based on the image, there is contradiction in the image. Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
        
        if args.gen_des!=True and args.gen_con:
            instruction = f"Based on the following contradiction: {contradiction} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
        if args.gen_des and args.gen_con!=True:
            instruction = f"Based on the following description: {sample['gen_des']} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
        
        if args.icot:
            #generate contradiction with des
            print("---------generate contradiction with cap--------")
            prompt=get_prompt(f'You are given an caption of an image and image: {sample["gen_des"]}. Based on the caption and image, return me the contradiction of between these 2 panels in two or three sentences.')
            print("[Prompt]:",prompt)

            inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)
            inputs = inputs.to(torch.float16)

            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result=processor.decode(output[0][2:], skip_special_tokens=True)

            if "assistant" in result:
                con_with_des = result.split('assistant')[1].strip()
            print('[Gen_con_with_des]:',con_with_des)
            sample['gen_con_with_des']=con_with_des

            #icot
            instruction = f"Based on the following contradiction and image: {con_with_des} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."


        print("[input]: ", instruction)
        prompt = get_prompt(instruction)
        inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)
        inputs = inputs.to(torch.float16)

        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        result=processor.decode(output[0][2:], skip_special_tokens=True)

        if "assistant" in result:
            pred = result.split('assistant')[1].strip()
            print('[output]:',pred)


            sample["input"] = instruction        
            sample["output"] = pred
        results.append(sample)
    
    with open(args.write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

if __name__=='__main__':
    main()