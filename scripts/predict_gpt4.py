import json
import os
import openai
from openai import OpenAI

import time
from nltk.tokenize import word_tokenize
import base64
from mimetypes import guess_type
import argparse
import requests
from PIL import Image
import json
import torch


def gpt4_vision_generation(input_prompt, model="gpt-4"):
    # Initialize the OpenAI API client
    client = OpenAI(api_key="")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": input_prompt
            }
        ]
    )
    return response.choices[0].message.content



def get_prompt(instruction):
    return instruction


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--gen_con', action= "store_true")
    parser.add_argument('--using_gen_des', action= "store_true")
    parser.add_argument('--gen_des_file', type=str, required=False)

    args = parser.parse_args()
    print(args)

    data = json.load(open(args.read_path))
    results = []
    if args.using_gen_des:
        print("---------Load gen description--------")
        with open(args.gen_des_file, "r") as f:
            gen_des_file = json.load(f)

    for sample in data:


        caption = sample["caption"]
        contradiction = None
        instruction = None  
        
        if args.using_gen_des:
            print("---------Use generated description--------")
            for g in gen_des_file:
                if g['image_file']==sample['image_file']:
                    caption=g['gen_des']
                    print('[Gen_caption]:',caption)
                    break

        if args.gen_con:
            instruction = f"Based on the {caption}, Tell me the contradiction between the two panels in 2 sentences."
            prompt = get_prompt(instruction)
            result = gpt4_vision_generation(prompt)
            contradiction = result
            print("[gen_con]:", contradiction)    
            sample["gen_con"] = contradiction

            
        #get question
        question =sample[args.task]

        #only img input
        if args.gen_con!=True:
            instruction = f"Based on the {caption}, there is contradiction in the caption. Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
        
        if args.gen_con:
            instruction = f"Based on the following contradiction: {contradiction} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
       
        if args.using_gen_des:
            instruction = f"Based on the following description: {caption} Tell me the best option in the following options who represents the deep semantics? \n{question} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."

        print("[input]: ", instruction)
        prompt = get_prompt(instruction)
        pred = gpt4_vision_generation(prompt)


        print('[output]:',pred)
        sample["input"] = instruction        
        sample["output"] = pred
        results.append(sample)
    
    with open(args.write_path, "w") as f_w:
        json.dump(results, f_w, indent=4, ensure_ascii=False)

if __name__=='__main__':
    main()