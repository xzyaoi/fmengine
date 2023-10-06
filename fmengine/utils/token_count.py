import os
import json
from multiprocessing import Pool
from transformers import AutoTokenizer

def count_tokens(args):
    return len(args[0].encode(args[1]))


def parallel_count(tokenizer_name, texts):
    # get cpu count
    n_processor = os.cpu_count()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    with Pool(n_processor) as pool:
        return sum(pool.map(count_tokens, [(tokenizer, text) for text in texts]))

def get_epoch_size(batch_size, seq_len, dp_degree, total_tokens):
    return int(total_tokens / (batch_size * seq_len * dp_degree))

def count(tokenizer, text):
    return len(tokenizer.encode(text))

def count_tokens_from_file(
    filename: str,
    tokenizer_name: str,
    field: str = 'text',
    special_tokens: dict = None, 
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if special_tokens is None:
        special_tokens = {}
    with open(filename, 'r') as f:
        data = [json.loads(x)[field] for x in f.readlines()]
    return sum([count(tokenizer, x) for x in data])
