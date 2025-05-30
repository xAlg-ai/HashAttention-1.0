import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from inf_llm.utils import GreedySearch # used for chunk based running of hte model for generating batches of training data
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc
import sys

MODELNAME = os.environ.get("MODEL", None)
if MODELNAME == "llama":
    from hashattention.hashattention_llama import convert_usa, load_usa, reset_usa, set_train_usa_mode, set_eval_usa_mode, print_stats
elif MODELNAME == "mistral":
    from hashattention.hashattention_mistral import convert_usa, load_usa, reset_usa, set_train_usa_mode, set_eval_usa_mode, print_stats
else:
    raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--conv_type", required=True)
    parser.add_argument("--train_datasets", type=str, nargs="+", default=None)
    parser.add_argument("--validation_dataset", type=str, default=None)
    
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--truncate_len", type=int, default=64000)
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_usa", type=str, default=None)
    parser.add_argument("--load_usa", type=str, default=None)
    parser.add_argument("--skip_first_examples", type=int, default=-1)

    #loss stuff
    parser.add_argument("--loss", type=str, default='bce')
    parser.add_argument("--bce_alpha", type=float, default=20.0)
    parser.add_argument("--bce_beta", type=float, default=0.)
    parser.add_argument('--usa_num_layers', type=int, default=3)
    parser.add_argument('--usa_final_dim', type=int, default=32)
    

    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.from_cli(extra_args)
    conf.model_path = args.model_path
    conf.conv_type = args.conv_type
    conf.verbose = args.verbose
    conf.limit = args.limit
    conf.epochs = args.epochs
    conf.save_usa = args.save_usa
    conf.load_usa = args.load_usa
    conf.truncate_len = args.truncate_len
    conf.skip_first_examples = args.skip_first_examples
    conf.loss = args.loss
    conf.bce_alpha = args.bce_alpha
    conf.bce_beta = args.bce_beta
    conf.usa_num_layers = args.usa_num_layers
    conf.usa_final_dim = args.usa_final_dim
    conf.train_datasets = args.train_datasets
    conf.validation_dataset = args.validation_dataset
    conf.chunk_size = args.chunk_size

    print(conf)
    return conf


def get_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda", attn_implementation="eager")
    config = AutoConfig.from_pretrained(args.model_path)

    config.lth_init_dim = 128
    config.lth_final_dim = args.usa_final_dim
    config.lth_thold = 0
    config.init_budget = 128
    config.heavy_budget = 0.125
    config.recent_budget = 128
    config.usa_retrieve_depth = 6
    config.usa_eval_mode = "simple"
    config.lth_num_layers = args.usa_num_layers
    usa_modules = load_usa(config, args.load_usa)
    usa_modules = usa_modules.bfloat16()
    model = convert_usa(model, config, usa_modules, collect_stats=True, train_usa=True)

    def get_bce_loss_function(alpha, beta):
        def loss_function(yhat ,ytarget):
            w = ytarget.shape[-1] * beta +  alpha
            weight = ytarget * (w - 1) + torch.ones_like(ytarget)
            loss = torch.nn.functional.binary_cross_entropy(yhat.reshape(-1), ytarget.reshape(-1), weight = weight.reshape(-1))
            return loss
        return loss_function
    optimizer = torch.optim.Adam(usa_modules.parameters(), lr = 0.001)

    if args.loss == 'bce':
        print("Using BCE loss with", args.bce_alpha, args.bce_beta)
        loss_function = get_bce_loss_function(args.bce_alpha, args.bce_beta)
    else:
        raise NotImplementedError
    set_train_usa_mode(model, loss_function, optimizer)
    print(model)

        
    return model, tokenizer, usa_modules

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.strip().lower()
    assert model_name in ["mistral-inst", "qwen", "minicpm", "llama-3-inst"]
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def train(
    args, model, tokenizer, 
    train_data, prompt_format,
    max_gen, gen_chunk_size,
    tr_truncate_len, 
    save_usa_path,
    usa_modules_ptr,
    skip_first_examples = -1,
    finetune=False
):
    searcher = GreedySearch(model, tokenizer)
    cur = 0

    text = ""
    text_len = 0
    char_to_token_factor = 5
    itr = 0

    for i, json_obj in tqdm(enumerate(train_data)):
        if i < skip_first_examples:
            continue
        gc.collect()
        if not finetune:
            text_len += len(json_obj['text'])
            text = text + "Passage: " + json_obj['text']
            if text_len < tr_truncate_len * char_to_token_factor:
                #accumulate more text
                continue

            prompt = prompt_format.format(text=text)
        else:
            prompt = prompt_format.format(**json_obj)
        # reset values
        text = ""
        text_len = 0

        extra_end_token_ids = []
        ## ASSERT MODEL IS LLAMA
        extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
        add_special_tokens = True
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]    
        print(tokenized_prompt.shape, len(text), flush=True)
        if tr_truncate_len is not None:
            tokenized_prompt = tokenized_prompt[:tr_truncate_len]
        searcher.clear()
        output = searcher.generate(
            input_ids = tokenized_prompt,
            max_length=max_gen,
            chunk_size=gen_chunk_size,
            extra_end_token_ids=extra_end_token_ids,
            prefetch_offset=1
        )
        itr += 1
        reset_usa(model) # removes the state if stored
        if itr % 10 == 0:
            if save_usa_path is not None:
                torch.save(usa_modules_ptr.cpu().state_dict(), save_usa_path)
                usa_modules_ptr = usa_modules_ptr.cuda()



def get_dataset(dataset):
    if dataset == "openwebtext":
        dataset =  load_dataset("Skylion007/openwebtext", trust_remote_code=True)
        data = dataset['train']
        dataset2prompt = "Use the following collection of passages to answer the questions. {text} "
        dataset2maxlen = 1024
    # benchmark datasets
    else:
        raise NotImplementedError
    return data, dataset2prompt, dataset2maxlen


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    model, tokenizer, usa_modules = get_model_and_tokenizer(args)

    train_datasets = args.train_datasets

    for epoch in range(args.epochs): # for USA training
        for train_dataset in train_datasets:
            train_data, prompt_format, maxlen = get_dataset(train_dataset)
            print(f"Train {train_dataset}")
            max_gen = 1
            train( args, model, tokenizer, 
                        train_data, prompt_format,
                        max_gen, args.chunk_size, args.truncate_len,
                        args.save_usa, usa_modules, args.skip_first_examples, 
                        train_dataset != "openwebtext"
                       )
