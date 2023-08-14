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
import math
import yaml

from tqdm import tqdm
from scipy.io import wavfile
from Levenshtein import ratio

from datasets import load_dataset
from datasets import Audio
from sacrebleu.metrics import BLEU
from whisper.tokenizer import get_tokenizer

from metrics import calc_avg_lagging
from metrics import calculate_avg_latency

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

language_dict = {'ja':'Japanese','ru':'Russian','es':'Spanish','de':'German','ar':'Arabic'}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="medium", help="Model to use", choices=["tiny", "base", "small", "medium"])
parser.add_argument("--policy", default="greedy", help="Policy to use", choices=["greedy", "offline", "confidence", "consensus"])
parser.add_argument("--num_examples", default=50, help="the number of examples to use for evaluation", type=int)
parser.add_argument("--consensus_threshold", default=0.9, help="If edit distance ratio > threshold then we speak the segment (for CP)", type=float)
parser.add_argument("--confidence_threshold", default=0.9, help="The threshold over which we speak the segment (for CAP)", type=float)
parser.add_argument("--api_keys_path", default="api_keys.yml", help="The path to the api keys file", type=str)
args = parser.parse_args()

with open(args.api_keys_path, "r") as f:
    try:
        keys = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

for lang in ['ja', 'es', 'ru', 'ar']:
    covost_2 = load_dataset("google/xtreme_s", f"covost2.{lang}.en")
    covost_2 = covost_2.cast_column("audio", Audio(sampling_rate=16000))

    model = whisper.load_model(args.model)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    translate_options = dict(task="translate", language=language_dict[lang], beam_size=5, best_of=5, fp16=torch.cuda.is_available())
    tokenizer = get_tokenizer(model.is_multilingual, language=language_dict[lang])
    
    for clip_increment in [1, 2]:
        full_begin = time.time()
        clip_lens = []
        ref_translations = []
        translations = []
        latencies = []
        ALs = []
        LAALs = []
        offline_ALs = []
        offline_LAALs = []
        offline_translations = []
        offline_latencies = []
        
        for entry in tqdm(covost_2["validation"].select(range(0,args.num_examples))):
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
                spoken_ids = []
                clip = audio[int(spoken*16000):min(int(progress * 16000), len(audio))]
                
                result = model.transcribe(clip, prompt=''.join(translation), **translate_options)
                #print(f"[[{[(s['text'], s['start'], s['end'], s['no_speech_prob']) for s in result['segments']]}]]")

                for s in result['segments']:
                                        
                    # If no speech probability is very high, don't speak regardless of policy
                    if s['no_speech_prob'] > 0.9:
                        continue
                    
                    # If greedy policy, speak immediately
                    if args.policy == "greedy":
                        translation.append(s['text'])
                        latency.append(progress + time.time() - start)
                        spoken += s['end']
                        continue
                    
                    # If confidence policy & confidence is high, speak
                    if args.policy == "confidence" and s['no_speech_prob'] < 1 - args.confidence_threshold:
                        translation.append(s['text'])
                        latency.append(progress + time.time() - start)
                        spoken += s['end']
                        continue
                    
                    # If consensus policy, try to find consensus
                    if args.policy == "consensus" and prev:
                        # If the segment is past the end of where previous had, we're done
                        if s['id'] >= len(prev['segments']):
                            break
                        
                        # Check for consensus and if so, speak
                        if ratio(s['text'], prev['segments'][s['id']]['text']) >= args.consensus_threshold:
                            translation.append(s['text'])
                            latency.append(progress + time.time() - start)
                            spoken += s['end']
                            spoken_ids.append(s['id'])
                        else:
                            break

                # If using consensus we must delete spoken segments from result before saving to prev
                if args.policy == "consensus" and result:
                    result['segments'] = [s for s in result['segments'] if s['id'] not in spoken_ids]
                    prev = result
            
            offline_start = time.time()
            offline_translation = model.transcribe(audio, **translate_options)['text']
            offline_latency = clip_len + time.time() - offline_start
            
            # If by the end of the clip we never translated anything, fall back to offline
            if len(translation) == 0:
                translation.append(offline_translation)
                latency.append(offline_latency)
            
            avg_latency = calculate_avg_latency(translation, latency, tokenizer)
            avg_lagging = calc_avg_lagging(translation, latency, tokenizer, entry["translation"], clip_len)
            length_adaptive_avg_lagging = calc_avg_lagging(translation, latency, tokenizer, entry["translation"], clip_len, is_len_adaptive=True)
            offline_AL = calc_avg_lagging(translation, [offline_latency]*len(translation), tokenizer, entry["translation"], clip_len)
            offline_LAAL = calc_avg_lagging(translation, [offline_latency]*len(translation), tokenizer, entry["translation"], clip_len, is_len_adaptive=True)
            
            offline_translations.append(offline_translation)
            offline_latencies.append(offline_latency)
            translations.append(''.join(translation))
            ALs.append(avg_lagging)
            LAALs.append(length_adaptive_avg_lagging)
            offline_ALs.append(offline_AL)
            offline_LAALs.append(offline_LAAL)
            latencies.append(avg_latency)
            ref_translations.append(entry["translation"])
            
        data = pd.DataFrame(dict(ref_translation=ref_translations, translation=translations, latency=latencies, AL=ALs, LAAL=LAALs,
                                    offline_translation=offline_translations, offline_latency=offline_latencies, offline_AL=offline_ALs, offline_LAAL=offline_LAALs))
        
        tgt_bleu = BLEU(trg_lang='en')

        mean_tr_bleu = tgt_bleu.corpus_score(list(data['translation']), [list(data['ref_translation'])])
        mean_latency = sum([l for l in latencies if l]) / float(len([l for l in latencies if l]))
        mean_AL = sum([l for l in ALs if l]) / float(len([l for l in ALs if l]))
        mean_LAAL = sum([l for l in LAALs if l]) / float(len([l for l in LAALs if l]))
        mean_tr_bleu_offline = tgt_bleu.corpus_score(list(data['offline_translation']), [list(data['ref_translation'])])
        mean_offline_latency = sum(offline_latencies) / float(len(offline_latencies))
        mean_offline_AL = sum([l for l in offline_ALs if l]) / float(len([l for l in offline_ALs if l]))
        mean_offline_LAAL = sum([l for l in offline_LAALs if l]) / float(len([l for l in offline_LAALs if l]))

        num_examples = len(covost_2["validation"])
        full_end = time.time()

        print(data[['ref_translation', 'translation', 'latency', 'AL','LAAL','offline_translation', 'offline_latency', 'offline_AL', 'offline_LAAL']].head())
        print(f"Translation BLEU Score for model size [{args.model}] with language [{lang}] and clip increment [{clip_increment}] is: {mean_tr_bleu.score}")
        print(f"Full Time Taken for model size [{args.model}] with language [{lang}] and clip increment [{clip_increment}] is: {full_end - full_begin}")
        
        if args.policy == "confidence":
            fname = f'./outputs/results-online-{time.time()}-{args.model}-{args.policy}-{lang}-{clip_increment}-{args.confidence_threshold}.txt'
        elif args.policy == "consensus":
            fname = f'./outputs/results-online-{time.time()}-{args.model}-{args.policy}-{lang}-{clip_increment}-{args.consensus_threshold}.txt'
        else:
            fname = f'./outputs/results-online-{time.time()}-{args.model}-{args.policy}-{lang}-{clip_increment}.txt'
            
        with open(fname, 'w+', encoding="utf-8") as out_f:
            out_f.write(str(data[['ref_translation', 'translation', 'latency', 'AL', 'LAAL', 'offline_translation', 'offline_latency', 'offline_AL', 'offline_LAAL']].head()) + "\n\n")
            out_f.write(f"Time taken: {full_end - full_begin} seconds\n")
            out_f.write(f"Num examples: {args.num_examples}\n")
            out_f.write(f"Online Translation BLEU Score: {mean_tr_bleu.score}\n")
            out_f.write(f"Online Latency: {mean_latency}\n")
            out_f.write(f"Online AL: {mean_AL}\n")
            out_f.write(f"Online LAAL: {mean_LAAL}\n")
            out_f.write(f"Offline Translation BLEU Score: {mean_tr_bleu_offline.score}\n")
            out_f.write(f"Offline Latency: {mean_offline_latency}\n")
            out_f.write(f"Offline AL: {mean_offline_AL}\n")
            out_f.write(f"Offline LAAL: {mean_offline_LAAL}\n")