import json
import random

def random_choice_from_file(file_path,choose_num=1000,new_path_suffix='random',data_type='huatuo'):
    with open(file_path, 'r',encoding="utf-8") as f:
        if data_type == 'huatuo':
            lines = f.readlines()
            random.shuffle(lines)
            choose_samples = lines[:choose_num]
        else:
            # for json object
            data = json.load(f)
            random.shuffle(data)
            choose_samples = data[:choose_num]
            # convert dict element to string
            choose_samples = [json.dumps(sample,ensure_ascii=False)+'\n' for sample in choose_samples]

        # add suffix to new file and keep json format
        file_path,file_type = file_path.split('.')
        new_path = file_path+'_'+new_path_suffix+ f'_{choose_num}' + '.'+file_type

        with open(new_path, 'w+',encoding="utf-8") as f_out:
            for line in choose_samples:
                f_out.write(line)

# split data to train and val
def split_data(input_file,out_dir,split_ratio=0.9):
    with open(input_file, 'r',encoding="utf-8") as f:
        lines = f.readlines()
        random.shuffle(lines)
        train_lines = lines[:int(len(lines)*split_ratio)]
        val_lines = lines[int(len(lines)*split_ratio):]
        with open(out_dir+'train.json', 'w+',encoding="utf-8") as f_train:
            for line in train_lines:
                f_train.write(line)
        with open(out_dir+'val.json', 'w+',encoding="utf-8") as f_val:
            for line in val_lines:
                f_val.write(line)


# convert another dict type to uniform dict type
def convert_dict_type(input_file,output_file,data_type='huatuo'):

    # standard format data
    #{"id": "1_gout_appointment", "name": "gout_appointment_instruction", "instruction": "我想预约看痛风，该挂什么科？", "instances": [{"input": "", "output": "您好，痛风一般属于风湿免疫科，建议您挂风湿免疫科的号进行就诊。"}]}

    format_data = []
    with open(input_file, 'r',encoding="utf-8") as f:
        origin_data = f.readlines()
        if data_type == 'huatuo':
            # "query" and "answer" convert to "instruction" and "output",'input' as null
            for line in origin_data:
                data = json.loads(line)
                standard_dict = {"instruction": "","instances": [{"input": "", "output": ""}]}
                standard_dict['instruction'] = data['query']
                standard_dict['instances'][0]['output'] = data['answer']
                format_data.append(standard_dict)
        elif data_type == 'general':
            for line in origin_data:
                data = json.loads(line)
                standard_dict = {"instruction": "","instances": [{"input": "", "output": ""}]}
                standard_dict['instruction'] = data['instruction']
                standard_dict['instances'][0]['input'] = data['input']
                standard_dict['instances'][0]['output'] = data['output']
                format_data.append(standard_dict)
        else:
            for line in origin_data:
                data = json.loads(line)
                standard_dict = {"instruction": "","instances": [{"input": "", "output": ""}]}
                standard_dict['instruction'] = data['instruction']
                standard_dict['instances'] = data['instances']
                format_data.append(standard_dict)
            
    return format_data



# concat three data files to final sft data file
def concat_data(med_data_file,general_data_file,generate_sft_raw_data_file,final_sft_data_file):
    
    total_lines = []
    
    with open(med_data_file, 'r',encoding="utf-8") as f:
        med_lines = f.readlines()
        # convert dict type to uniform dict type
        med_lines = convert_dict_type(med_data_file,med_lines,data_type='huatuo')
        total_lines.extend(med_lines)
    with open(general_data_file, 'r',encoding="utf-8") as f:
        general_lines = f.readlines()
        # convert dict type to uniform dict type
        general_lines = convert_dict_type(general_data_file,general_lines,data_type='general')
        total_lines.extend(general_lines)
    with open(generate_sft_raw_data_file, 'r',encoding="utf-8") as f:
        generate_lines = f.readlines()
        # convert dict type to uniform dict type
        generate_lines = convert_dict_type(generate_sft_raw_data_file,generate_lines,data_type='sft')
        total_lines.extend(generate_lines)

    # shuffle data
    random.shuffle(total_lines)
    with open(final_sft_data_file, 'w+',encoding="utf-8") as f:
        for line in total_lines:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')
    
    


if __name__ == '__main__':

    med_data_file = 'data/public/huatuo-GPT-226k.jsonl'
    # random choice from hua tuo data
    # random_choice_from_file(med_data_file,choose_num=350,new_path_suffix='random')

    general_data_file = 'data/public/alpaca_data_zh_51k.json'
    # random choice from general data
    # random_choice_from_file(general_data_file,choose_num=500,new_path_suffix='random',data_type='general')


    # concat data
    generate_sft_raw_data_file = 'data/output/sft_data_raw.jsonl'
    med_data_file = 'data/public/huatuo-GPT-226k_random_350.jsonl'
    general_data_file = 'data/public/alpaca_data_zh_51k_random_500.json'
    final_sft_data_file = 'data/output/sft_data_1500.jsonl'
    concat_data(med_data_file,general_data_file,generate_sft_raw_data_file,final_sft_data_file)

    # # split data to train and val
    # out_dir = 'data/'
    # split_data(output_file,out_dir,split_ratio=0.9)