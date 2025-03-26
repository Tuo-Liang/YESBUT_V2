import json
import random

random.seed(1234)


def formulate_instruction(sample_dict, caption=None, task="contradiction",num=2):
    if task == "contradiction":
        instruction = formulate_instruction_contradiction_generation(caption,num)
    elif task == "moral_mcq":
        moral_mcq = sample_dict["moral_mcq"]
        instruction = formulate_instruction_moral_mcq(moral_mcq, caption,num)
    elif task == "title_mcq":
        title_mcq = sample_dict["title_mcq"]
        instruction = formulate_instruction_title_mcq(title_mcq, caption,num)
    else:
        print(f"[TASK ERROR!!!] {task}")
        return None
    return instruction

def gen_description(num):
    if num==0:
        instruction = f'''
The given comic shows the same situation from two opposite sides with contradictions. Write a one-paragraph literal description to describe the narrative of the comic.
'''
    if num==1:
        instruction = f'''
Please literally describe the context of the image in detail.
'''
    if num==2:
        instruction = f'''
Give me a detailed literally description of the image.
'''
    if num==3:
        instruction=f'''Describe this image specifically.'''
    return instruction


def formulate_instruction_contradiction_generation(caption=None,num=2):
    if caption is None: # no caption setting
        if num==0:
            instruction = f'''
The given comic shows the same situation from two opposite sides with contradictions.
Write a short explanation to illustrate the contradiction of the two sides.
'''     
        if num==1:
            instruction = f'''
Analyze the provided image, which is divided into two or more panels, each illustrating contrasting views of the same scenario. Describe the elements visible in each panel. Then concisely interpret how these elements convey contrasting perspectives in one or two sentences. Focus and only output the contradiction.
'''
        if num==2:
            instruction = f'''
Given an image, the image is divided into two or more panels. There is the contrast relationship in the image through panels. Describe the elements visible in each panel. Give me the concise interpretation how these panels convey contrasting perspectives, which you only need to output the contradiction in one or two sentences.
'''
        if num == 3:
            instruction = f'''Tell me the contradiction between the two panels in 2 sentences.'''
    else:
        if num==0:
            instruction = f'''
The given comic shows the same situation from two opposite sides with contradictions.
The literal caption of the comic is: {caption}
Write a short explanation to illustrate the contradiction of the two sides.
'''
        if num==1:
            instruction = f'''
Analyze the provided image with the following description: {caption}. Identify and concisely describe the contradiction depicted in the image in one or two sentences.
'''
        if num==2:
            instruction = f'''
      Based on the following image's description: {caption}. Give me the concise contradiction depicted in the image in one or two sentences.
'''
        if num == 3:
            instruction = f'''Based on the following image's description: {caption}. Tell me the contradiction between the two panels in 2 sentences.'''  
    return instruction


def formulate_instruction_moral_mcq(moral_mcq, caption=None,num=2):
    #caption=None
    if caption is None: # no caption setting
        if num==0:
            instruction = f'''
The given comic shows the same situation from two opposite sides with contradictions.
Which of the following options best represents the underlying philosophy of the comic?
{moral_mcq}
\nJust output the choice:
'''
        if num==1:
            instruction = f'''
        You are presented with an image, which is divided into two or more panels, each illustrating contrasting views of the same scenario. \nWhich of the following options best represents the philosophy of the image provided? \n{moral_mcq} \nSelect the correct option by typing the corresponding letter (A, B, C, or D).
'''
        if num==2:
            instruction = f'''
        Given an image, which has two or more panels. There is contrast in these panels. \nTell me the best option in the following options who represents the deep semantic of the image? \n{moral_mcq} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation.
'''
        if num == 3:
            instruction = f'''Based on the image, there is contradiction in the image. Tell me the best option in the following options who represents the deep semantics? \n{moral_mcq} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation.'''

    else:
        if num==0:
            instruction = f'''
The given comic shows the same situation from two opposite sides with contradictions.
with the following text: {caption}
Which of the following options best represents the underlying philosophy of the comic?
{moral_mcq}
\nJust output the choice:
''' 
        if num==1:
            instruction = f'''
        You are presented with an image. With the following text: {caption}. \nWhich of the following options best represents the philosophy of the image provided? \n{moral_mcq} \nSelect the correct option by typing the corresponding letter (A, B, C, or D).
'''
        if num==2:
            instruction = f'''
        Given an image. With the following text: {caption}. \nTell me the best option in the following options who represents the deep semantic of the image? \n{moral_mcq} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation.
'''
            
        if num == 3:
            instruction =f'''"Based on the following description: {caption} Tell me the best option in the following options who represents the deep semantics? \n{moral_mcq} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation.'''
    return instruction


def formulate_instruction_title_mcq(title_mcq, caption=None,num=2):
    #caption=None
    if caption is None: # no caption setting
        if num==0:
            instruction = f'''
The given comic shows the same situation from two opposite sides with contradictions.
Which of the following titles are the most suitable for the comic?
{title_mcq}
\nJust output the choice:
'''
        if num==1:
            instruction = f'''
You are presented with an image, which is divided into two or more panels, each illustrating contrasting views of the same scenario. \nWhich of the following title options best represents the image provided? \n{title_mcq} \nSelect the correct option by typing the corresponding letter (A, B, C, or D).
'''
        if num==2:
            instruction = f'''
You are presented with an image, which is divided into two or more panels, each illustrating contrasting views of the same scenario. \nWhich of the following title options best represents the image provided? \n{title_mcq} \nSelect the correct option by typing the corresponding letter (A, B, C, or D).
'''
        if num ==3:
            instruction = f'''Based on the image, there is contradiction in the image. Tell me the best option in the following options who represents the deep semantics? \n{title_mcq} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation.'''
    else:
        if num==0:
            instruction = f'''
The given comic shows the same situation from two opposite sides with contradictions.
With the following text: : {caption}
Which of the following titles are the most suitable for the comic?
{title_mcq}
\nJust output the choice:
'''     
        if num==1:
            instruction = f'''
        You are presented with an image. With the following text: {caption}. \nWhich of the following title options best represents the image provided? \n{title_mcq} \nSelect the correct option by typing the corresponding letter (A, B, C, or D).
'''
        if num==2:
            instruction = f'''
        Given an image. With the following text: {caption}. There is the contrast relationship in the image through panels. \nTell me the best title in the following title options who represents the image? \n{title_mcq} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation.
'''
        if num == 3:
            instruction =f'''"Based on the following description: {caption} Tell me the best option in the following options who represents the deep semantics? \n{title_mcq} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation.'''
    return instruction