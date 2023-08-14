import whisper

'''
Compute the average raw latency
'''
def calculate_avg_latency(translations, latencies, tokenizer):
    size = 0.0
    total_latency = 0.0
    for text, latency in zip(translations, latencies):
        num_tokens = float(len(tokenizer.encode(text)))
        size += num_tokens
        total_latency += latency * num_tokens
    if size == 0.0:
        return None
    return total_latency / size

'''
Compute the Average Lagging metric https://aclanthology.org/P19-1289.pdf
'''
def calc_avg_lagging(translations, latencies, tokenizer, reference, clip_len, is_len_adaptive=False):
    # Tokenize the translation and reference
    tok_translation = tokenizer.encode(''.join(translations))
    tok_reference = tokenizer.encode(reference) 
    
    # Scale clip length and latencies to be in ms and not seconds
    clip_len *= 1000.0
    latencies = [l*1000.0 for l in latencies]
    
    # If either translation or reference is blank, return None
    if len(tok_reference) == 0 or len(tok_translation) == 0:
        return None
    
    # If we're doing length-adaptive average lagging (https://aclanthology.org/2022.autosimtrans-1.2.pdf)
    if is_len_adaptive:
        ref_length = max(len(tok_translation), len(tok_reference))
    else:
        ref_length = len(tok_reference)
        
    # Calculate the latency for each token in the segments
    token_latencies = []
    for text, latency in zip(translations, latencies):
        num_tokens = len(tokenizer.encode(text))
        token_latencies.extend([latency]*num_tokens)
    
    # Find the idx of the first target token generated after the model has seen the full input
    first_target_idx = len(tok_translation) - 1
    for i, l in enumerate(token_latencies):
        if l > clip_len:
            first_target_idx = i
            break
    
    # Compute average lagging
    lagging = 0.0
    for i in range(first_target_idx + 1):
        lagging += token_latencies[i] - (clip_len / ref_length) * (i - 1)
    
    return lagging / (first_target_idx + 1)