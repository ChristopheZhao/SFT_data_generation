from unsloth import FastLanguageModel
import torch

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""



# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

# load model
def load_model(model_name = "unsloth/llama-3-8b-bnb-4bit"):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    return model, tokenizer

# add LoRA adapters so we only need to update 1 to 10% of all parameter
def lora_adapt(base_model):

    model = FastLanguageModel.get_peft_model(
    base_model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    )

    return model

# format prompt for customized dataset
def formatting_sft_prompts_func(examples):
    
    instructions = examples["instruction"]
    
    instances = examples['instances']
    inputs = []
    outputs = []

    for instance in instances:
        inputs.append(instance[0]['input'])
        outputs.append(instance[0]['output'])
    texts = []
    
    for instruction, input, output in zip(instructions, inputs, outputs):
        # print(f'instruction = {instruction}\ninput = {input}\noutput = {output}')
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


# load customize dataset
def load_data(dataset_path = "/content/train_sft", split = 'train'):
    dataset = load_dataset(path=dataset_path,split =split)
    dataset = dataset.map(formatting_sft_prompts_func, batched = True,)

    return dataset


# initialize trainer
def init_trainer(model, tokenizer, dataset, max_seq_length = 2048):
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
    )

    return trainer

# save model
def save_model(model, tokenizer, path = "outputs"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_model(model_path = "outputs"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    return model, tokenizer

# inference
def inference(model, tokenizer,intruction='Continue the fibonnaci sequence', input='1, 1, 2, 3, 5, 8'):
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            intruction, # instruction
            input, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    res = tokenizer.batch_decode(outputs)
    return res


def train_pipeline(ori_model_path,data_path,output_path = "./data/output/lora_ckpt"):
    # load model
    model, tokenizer = load_model(ori_model_path)

    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token # assign EOS_TOKEN

    # lora adapt
    model = lora_adapt(model)
    # load dataset
    dataset = load_data(data_path)
    
    # initialize trainer
    trainer = init_trainer(model, tokenizer, dataset)
    # train model
    trainer.train()

    print("Training finished!")

    # save model
    save_model(model, tokenizer, path = output_path)

    print("loRa model saved!")


if __name__ == '__main__':

    ori_model_path = "/root/data/llama-3-8b-bnb-4bit"
    data_path = "/root/code/LLM/unsloth/sft_data"
    save_path = "/root/code/LLM/unsloth/output"
    # sft train
    train_pipeline(ori_model_path,data_path,output_path = save_path)

    # inference
    # model, tokenizer = load_model(model_path = save_path)
    # task_case = {'instruction':'Continue the fibonnaci sequence','input':'1, 1, 2, 3, 5, 8'}
    # res = inference(model, tokenizer,intruction=task_case['instruction'], input=task_case['input'])
    # print(res)

    # infrence use cmd input
    while 1:
        task_case = input("Please input your task case:")
        task_case = {"instruction":task_case,'input':''}
        res = inference(model, tokenizer,intruction=task_case['instruction'], input=task_case['input'])
        print(res)





