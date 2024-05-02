import re
import random
from tools.qianfan_requestor import init_qianfan_requestor
from config import config_dict
import json
from tqdm import tqdm
from template.disease_classification import disease_dictionary

cut_space = re.compile(r'[\s\'\n]+|```|json')

def format_prompt(prompt, instruction, disease_pro_list,instruct_tag='<sample_list>',personal_tag='<disease_list>'):
    
    prompt = cut_space.sub('', prompt)
    # replace instruction tag with instruction
    prompt = prompt.replace(instruct_tag, instruction)
    # replace specific disease tag with disease proposals
    prompt = prompt.replace(personal_tag, disease_pro_list)
    return prompt

def choose_disease_proposal(disease_dict,choose_num=3):
    disease_list = []
    for disease in disease_dict.keys():
        disease_list.extend(disease_dict[disease])
    
    return random.sample(disease_list,choose_num)


def generate_sft_data(instruction_file, prompt_file,output_file,chat_requestor,random_seed_num=3,generate_times=20):

    instruction_list = []
    personal_info_list = []

    with open(instruction_file, 'r',encoding="utf-8") as f:
        
        for line in f:
            # cut space for each line            
            instruction_line = cut_space.sub('', line.strip())
            instruction_list.append(instruction_line)

    # load prompt from file
    with open(prompt_file, 'r',encoding="utf-8") as f:
        prompt = f.read()

    with open(output_file, 'w+',encoding="utf-8") as f:

        for i in tqdm(range(generate_times)):
        
            # choose random instruction and disease proposal
            instruction = random.sample(instruction_list,random_seed_num)
            disease_pro_list = choose_disease_proposal(disease_dictionary,choose_num=random_seed_num)
            # concat instruction and personal info to prompt
            # there is only one shot for each instruction due the limitation nums of seed, if you have more seeds, you can add more samples in the context
            for j in range(random_seed_num):
                f_prompt = format_prompt(prompt, instruction[j], disease_pro_list[j])
                res = chat_requestor.send_message(f_prompt)
                print('api response:',res)
                try: gene_sample_list = json.loads(cut_space.sub('', res['result']))
                except Exception as e:
                    print('json loads error,continue')
                    continue

                for gene_sample in gene_sample_list:
                    gene_sample_str = json.dumps(gene_sample,ensure_ascii=False)
                    f.write(gene_sample_str+'\n')
                    f.flush()

                

if __name__ == '__main__':

    instruction_file = 'seed_tasks.jsonl'
    prompt_file = './template/prompt_template.txt'
    output_file = './data/output/sft_data_raw_1.jsonl'

    # initial baidu_ai_chat
    baidu_chat = init_qianfan_requestor(config_dict)

    generate_sft_data(instruction_file, prompt_file,output_file,baidu_chat,generate_times=70)

    