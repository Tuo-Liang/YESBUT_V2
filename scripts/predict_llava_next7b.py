import requests
from PIL import Image
import json
import torch
import os
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig


# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="int8",
    bnb_8bit_compute_dtype=torch.float16,
)


model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
#model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
#model_id = "llava-hf/llava-v1.6-34b-hf"

#model_id = "llava-hf/llava-next-72b-hf"
#model_id = "llava-hf/llava-next-110b-hf"


#device_map = {'cuda:0': 'cuda:0'}
#set environment cuda to 0



quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="int8",
)



model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    low_cpu_mem_usage=True,
    quantization_config=quantization_config, 
    use_flash_attention_2=True, 
    #device_map='auto'
)
processor = AutoProcessor.from_pretrained(model_id)

device='cuda'


def get_prompt(instruction):
    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": instruction},
          {"type": "image"},
        ],
    },
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
        image_path = args.image_folder + "/" + image_file
        img = Image.open(image_path)

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

            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result=processor.decode(output[0][2:], skip_special_tokens=True)

            if "ASSISTANT:" in result:
                cur_caption = result.split('ASSISTANT:')[1].strip()
            print('[Gen_caption]:',cur_caption)
            sample['gen_des']=cur_caption


        if args.gen_con and args.icot!=True:
            print("---------generate contradiction--------")
            prompt=get_prompt("Tell me the contradiction between the two panels in 2 sentences.")

            print("[Prompt]:",prompt)
            inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)

            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result=processor.decode(output[0][2:], skip_special_tokens=True)

            if "ASSISTANT:" in result:
                contradiction = result.split('ASSISTANT:')[1].strip()

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

            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result=processor.decode(output[0][2:], skip_special_tokens=True)

            if "ASSISTANT:" in result:
                con_with_des = result.split('ASSISTANT:')[1].strip()
            print('[Gen_con_with_des]:',con_with_des)
            sample['gen_con_with_des']=con_with_des

            #icot
            instruction = f"Based on the following contradiction and image: {con_with_des} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."


        print("[input]: ", instruction)
        prompt = get_prompt(instruction)
        inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)

        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        result=processor.decode(output[0][2:], skip_special_tokens=True)


        if "ASSISTANT:" in result:
            pred = result.split('ASSISTANT:')[1].strip()
            print('[output]:',pred)


            sample["input"] = instruction        
            sample["output"] = pred
        results.append(sample)
    
    with open(args.write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

if __name__=='__main__':
    main()