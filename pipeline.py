import io
import os
import numpy as np
import argparse

import torch
import pandas as pd
import urllib
import tarfile
import whisper
import torchaudio
import time

from tqdm import tqdm
from scipy.io import wavfile
from Levenshtein import ratio

from datasets import load_dataset
from datasets import Audio
from sacrebleu.metrics import BLEU
from whisper.tokenizer import get_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

language_dict = {'ja':'Japanese','ru':'Russian','es':'Spanish','de':'German','ar':'Arabic'}
tolerance = 0.2
no_consensus = True
consensus_threshold = 0.75

def calculate_avg_latency(translations, latencies, tokenizer):
    size = 0.0
    total_latency = 0.0
    for text, latency in zip(translations, latencies):
        num_tokens = float(len(tokenizer.encode(text)))
        size += num_tokens
        total_latency += latency * num_tokens
    return total_latency / size

for lang in ['ja', 'es', 'ru', 'ar']:
    covost_2 = load_dataset("google/xtreme_s", f"covost2.{lang}.en")
    covost_2 = covost_2.cast_column("audio", Audio(sampling_rate=16000))

    for size in ["tiny", "base", "small", "medium"]:
        model = whisper.load_model(size)
        print(
            f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )

        translate_options = dict(task="translate", language=language_dict[lang], beam_size=5, best_of=5, fp16=torch.cuda.is_available())
        tokenizer = get_tokenizer(model.is_multilingual, language=language_dict[lang])
        
        for clip_increment in [0.5, 1, 1.5, 2]:
            full_begin = time.time()
            ref_translations = []
            translations = []
            latencies = []
            offline_translations = []
            offline_latencies = []
            
            for entry in tqdm(covost_2["validation"].select(range(0,5))):
                audio = entry["audio"]["array"].astype(np.float32)
                clip_len = len(audio) / 16000
                progress = 0.0
                spoken = 0.0
                translation = []
                latency = []
                prev = None
                
                while progress < clip_len:
                    start = time.time()
                    progress += clip_increment
                    clip = audio[int(spoken*16000):min(int(progress * 16000), len(audio))]
                    
                    result = model.transcribe(clip, prompt=''.join(translation), **translate_options)
                    #print(f"[[{[(s['text'], s['start'], s['end'], s['no_speech_prob']) for s in result['segments']]}]]")
                    
                    # If there was a previous result, try to find consensus
                    for s in result['segments']:
                        
                        # If the segment is first and no speech prob is high, don't speak
                        if s['no_speech_prob'] > 0.1:
                            continue
                        
                        # If we are not using consensus just speak whatever we have immediately
                        if no_consensus:
                            translation.append(s['text'])
                            latency.append(progress + time.time() - start)
                            spoken += s['end']
                            continue
                        
                        if prev:
                            # If the segment is past the end of where previous had, we're done
                            if s['id'] >= len(prev['segments']):
                                break
                            
                            # Check for consensus and if so, speak
                            if ratio(s['text'], prev['segments'][s['id']]['text']) > consensus_threshold:
                                translation.append(s['text'])
                                latency.append(progress + time.time() - start)
                                spoken = s['end']
                            else:
                                break

                    # Save the current result to prev
                    prev = result
                    
                    # This is the last time we received new audio data from the queue.
                    phrase_time = time.time()
                
                if prev:
                    for s in prev['segments']:
                        if s['start'] > spoken - tolerance:
                            if s['start'] == 0.0 and s['no_speech_prob'] > 0.5:
                                continue
                            else:
                                translation.append(s['text'])
                                latency.append(progress + time.time() - start)
                
                offline_translations.append(model.transcribe(audio, **translate_options)['text'])
                offline_latencies.append(progress + time.time() - start)
                translations.append(''.join(translation))
                avg_latency = calculate_avg_latency(translation, latency, tokenizer)
                latencies.append(avg_latency)
                ref_translations.append(entry["translation"])
                
            data = pd.DataFrame(dict(ref_translation=ref_translations, translation=translations, latency=latencies, offline_translation=offline_translations, offline_latency=offline_latencies))
            
            tgt_bleu = BLEU(trg_lang='en')

            mean_tr_bleu = tgt_bleu.corpus_score(list(data['translation']), [list(data['ref_translation'])])
            mean_latency = sum(latencies) / float(len(latencies))
            mean_tr_bleu_offline = tgt_bleu.corpus_score(list(data['offline_translation']), [list(data['ref_translation'])])
            mean_offline_latency = sum(offline_latencies) / float(len(offline_latencies))

            num_examples = len(covost_2["validation"])
            full_end = time.time()

            print(data[['ref_translation', 'translation', 'latency', 'offline_translation', 'offline_latency']].head())
            print(f"Translation BLEU Score for model size [{size}] with language [{lang}] and clip increment [{clip_increment}] is: {mean_tr_bleu.score}")
            print(f"Full Time Taken for model size [{size}] with language [{lang}] and clip increment [{clip_increment}] is: {full_end - full_begin}")
            
            with open(f'./outputs/results-online-{time.time()}-{size}-{lang}-{clip_increment}.txt', 'w+', encoding="utf-8") as out_f:
                out_f.write(str(data[['ref_translation', 'translation', 'latency', 'offline_translation', 'offline_latency']].head()) + "\n\n")
                out_f.write(f"Time taken: {full_end - full_begin} seconds\n")
                out_f.write(f"Num examples: {60}\n")
                out_f.write(f"Online Translation BLEU Score: {mean_tr_bleu.score}\n")
                out_f.write(f"Online Latency: {mean_latency}\n")
                out_f.write(f"Offline Translation BLEU Score: {mean_tr_bleu_offline.score}\n")
                out_f.write(f"Offline Latency: {mean_offline_latency}\n")