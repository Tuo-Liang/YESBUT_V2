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
import re
import ast

API_KEY='your keys'
client = OpenAI(api_key=API_KEY)
def openai_generate(input_prompt, model="gpt-3.5-turbo-0125", temperature=1):

    if model == "chatgpt":
        model = "gpt-3.5-turbo"
    elif model == "gpt4":
        response = "gpt-4"

    for _ in range(5):
        try:
            response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": input_prompt}
                                ],
                            }
                        ],
                        temperature=temperature,
                    )
            break
        except Exception as e:
            print(["[OPENAI ERROR]: ", e])
            if "encountered an issue with repetitive patterns in your prompt" in e._message:
                return None
            response = None
            time.sleep(5)
    if response != None:
    # print(response)
        response = response.choices[0].message.content
    return response

def eval_one_caption(ref, gen):
    prompt = f'''
- Candidate literal description:
{gen}

- Reference literal description:
{ref}

Task: You need to determine how accurately the above candidate literal description matches the given reference literal description of a comic narrative.

Using a scale from 1 to 5, rate the accuracy with which the candidate description matches the reference description, with 1 being the least accurate and 5 being the most accurate.
Please directly output a score by strictly following this format: \"[[score]]\", for example: \"Rating: [[3]]\".
'''
    print(f"\n[log-input]: {[prompt]}")
    judgment = openai_generate(prompt.strip())
    #print(f"[log-output]: {[judgment]}")
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    # print("[log-llm_judge_eval-judgement]: ", [judgment])
    if judgment == None:
        return 0
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = 0
    print(f"\n[log-output]: {[rating]}")
    return rating


def evaluate_caption(read_path):
    data = json.load(open(read_path))
    score_list = []
    for sample in data:
        ref = sample["caption"]
        gen = sample["gen_des"]
        # if "output" not in sample:
        #     gen = sample["gen_des"]
        # else:
        #     gen = sample["output"]
        # if "000000000000000" in gen or ".................." in gen:
        #     gen = gen.replace("000000000000000", "")
        #     gen = gen.replace("..................", "")
        score = eval_one_caption(ref, gen)
        score_list.append(score)
    return score_list


def eval_one(caption, ref, gen):
    prompt = f'''
Background: You are an impartial judge. You will be given a literal description of a comic that presents the same situation from two opposing perspectives, highlighting contradictions. You will also be provided with a gold-standard illustration as reference that effectively demonstrates these narrative contradictions.

Your task is to evaluate the quality of a generated illustration and determine whether it accurately depicts the narrative contradictions in the comic. Then, assign a score on a scale of 1 to 5, where 1 is the lowest and 5 is the highest, based on its quality.

- The literal description of the comic:
{caption}

- The reference contradiction illustration:
{ref}

- The generated contradiction illustration:
{gen}

Please directly output a score by strictly following this format: \"[[score]]\", for example: \"Rating: [[3]]\".
'''
    print("[log-input]: ", prompt)
    judgment = openai_generate(prompt)
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    if judgment is None:
        return 0
    # print("[log-llm_judge_eval-judgement]: ", [judgment])
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = 0
    print(f"\n[log-output]: {[rating]}")
    return rating

def evaluate_contradiction(read_path):
    data = json.load(open(read_path))
    score_list = []
    for sample in data:
        caption = sample["caption"]
        ref = sample["contradiction"]
        gen = sample["gen_con"]
        if gen is None:
            continue

        # if "000000000000000" in gen or ".................." in gen:
        #     gen = gen.replace("000000000000000", "")
        #     gen = gen.replace("..................", "")
        score = eval_one(caption, ref, gen)
        score_list.append(score)
    return score_list


#
if __name__ =='__main__':
    folder='con'
    save_folder='g_eval'
    
    for file in os.listdir(folder):
        if file[-8:-5]=='des':
            print(file)
            cap_list = evaluate_caption(os.path.join(folder, file))
            #save the score_list
            name = file.split('.')[0]
            print('caption scoring')
            with open(f'{save_folder}/{name}_des_score.json', 'w',encoding='utf-8') as f:
                json.dump(cap_list, f)

        if file[-8:-5]=='con':
            print(file)
            print('contradiction scoring')
            con_list = evaluate_contradiction(os.path.join(folder, file))
            #save the score_list
            name = file.split('.')[0]

            with open(f'{save_folder}/{name}_con_score.json', 'w',encoding='utf-8') as f:
                json.dump(con_list, f)
    
    print('done')
        