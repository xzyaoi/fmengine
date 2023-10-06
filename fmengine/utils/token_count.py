import os
from multiprocessing import Pool
from transformers import AutoTokenizer

def count_tokens(args):
    return len(args[0].encode(args[1]))

def parallel_count(tokenizer_name, texts):
    n_processor = os.cpu_count()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    with Pool(n_processor) as pool:
        return sum(pool.map(count_tokens, [(tokenizer, text) for text in texts]))

def get_epoch_size(batch_size, seq_len, dp_degree, total_tokens):
    return int(total_tokens / (batch_size * seq_len * dp_degree))