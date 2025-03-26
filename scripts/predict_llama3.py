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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.3-70B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # You can also use 'fp4' depending on your requirements
    bnb_4bit_compute_dtype=torch.bfloat16  # Ensure your hardware supports bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_id)

def llama3_predict(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(**input_ids, max_new_tokens=512)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result





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
            result= llama3_predict(prompt)
            if "assistant" in result:
                contradiction = result.split('assistant')[-1].strip()

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

            result=llama3_predict(prompt)

            con_with_des = result
            print('[Gen_con_with_des]:',con_with_des)
            sample['gen_con_with_des']=con_with_des

            #icot
            instruction = f"Based on the following contradiction and image: {con_with_des} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."


        print("[input]: ", instruction)
        prompt = get_prompt(instruction)
        result=llama3_predict(prompt)

        if "assistant" in result:
            pred = result.split('assistant')[-1].strip()
            print('[output]:',pred)


            sample["input"] = instruction        
            sample["output"] = pred
            results.append(sample)
    
    with open(args.write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

if __name__=='__main__':
    main()