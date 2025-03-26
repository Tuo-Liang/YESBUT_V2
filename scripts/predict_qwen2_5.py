from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, AutoConfig, AutoModelForImageTextToText,AutoModelForCausalLM
import requests
from PIL import Image
import json
from prompts import *




# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_quant_type="int8",
# )


device="cuda"

model_id ="Qwen/Qwen2.5-72B-Instruct-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",).to(device)  


def qwen2_5_predict(text):
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def get_prompt(instruction):

    conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": instruction},
        ]

    # Preprocess the inputs
    text_prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True)

    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
    return text_prompt




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--gen_des', action= "store_true")
    parser.add_argument("--gt_des",action="store_true")
    parser.add_argument('--gen_con', action= "store_true")
    parser.add_argument('--with_social_info', action= "store_true")
    parser.add_argument('--icot', action= "store_true")
    args = parser.parse_args()
    print(args)

    data = json.load(open(args.read_path))
    results = []
    

    for sample in data:
        #transfer the image to url
        cur_caption =None
        contradiction = None
        instruction = None  


        if args.gt_des:
            print("---------Use ground truth caption--------")
            cur_caption=sample['caption']
            print('[GT_caption]:',cur_caption)


        if args.gen_con:
            print("---------generate contradiction--------")
            instruction = f'''Based on the descripiton of the comic: {cur_caption} Tell me the contradiction between the two panels in 2 sentences.'''
            prompt=get_prompt(instruction)

            print("[Prompt]:",prompt)
            result= qwen2_5_predict(prompt)
            contradiction =result

            print('[Gen_contradiction]:',contradiction)
            sample['gen_con']=contradiction
        
        if args.with_social_info:
            print("---------Use social info--------")
            social_info=sample['social_info']
            culture_info=sample['cultural_bg']
            print('[Social_info]:',social_info,culture_info)

            
        #get question
        question =sample[args.task]

        #only img input
        
        if args.gt_des and not args.gen_con:
            instruction = f"Based on the following description: {cur_caption} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
        
        if args.gen_con:
            instruction = f"Based on the following contradiction: {contradiction} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
        
        if args.with_social_info:
            instruction = f"Based on the image and following social information: {social_info} And culture background: {culture_info} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation." 

        if args.icot:
            #generate contradiction with des
            print("---------generate contradiction with cap--------")
            prompt=get_prompt(f'You are given an caption of an image and image: {sample["gen_des"]}. Based on the caption and image, return me the contradiction of between these 2 panels in two or three sentences.')
            print("[Prompt]:",prompt)

            result=qwen2_5_predict(prompt)

            con_with_des = result
            print('[Gen_con_with_des]:',con_with_des)
            sample['gen_con_with_des']=con_with_des

            #icot
            instruction = f"Based on the following contradiction and image: {con_with_des} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."


        print("[input]: ", instruction)
        prompt = get_prompt(instruction)
        result=qwen2_5_predict(prompt)

        pred = result
        print('[output]:',pred)


        sample["input"] = instruction        
        sample["output"] = pred
        results.append(sample)
    
    with open(args.write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

if __name__=='__main__':
    main()