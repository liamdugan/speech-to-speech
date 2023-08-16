from multiprocessing import Process, Queue
import requests, json, argparse, yaml
import whisper
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
from playsound import playsound
from Levenshtein import ratio
from recorder import MicrophoneRecorder
from translator import WhisperLocalTranslator, WhisperAPITranslator

parser = argparse.ArgumentParser()
parser.add_argument("--microphone_id", default=1, help="ID for the input microphone", type=int)
parser.add_argument("--verbose", action='store_true', help='Whether to print out the intermediate results or not')
parser.add_argument("--use_local", action='store_true', help='Whether to use the local whisper model instead of the API')
parser.add_argument("--api_keys_path", default="api_keys.yml", help="The path to the api keys file", type=str)
parser.add_argument("--config", default="config.yml", help="The path to the config file", type=str)
args = parser.parse_args()

# Retrieve the API keys from the API keys file
with open(args.api_keys_path, "r") as f:
    keys = yaml.load(f, Loader=yaml.loader.SafeLoader)

# Retrieve the Config info from the config file
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    
def recognize(q):
    phrase_time = None
    spoken = 0.0
    prev = None

    # Initialize the Translator
    if args.use_local:
        translator = WhisperLocalTranslator(config['model_name'], config['input_language'])
    else:
        translator = WhisperAPITranslator(keys["openai_api_key"], keys["openai_org_id"])
    
    # Initialize the Recorder
    recorder = MicrophoneRecorder(args.microphone_id, config['energy_threshold'], config['sampling_rate'])
    
    # Start recording and cue the user that we're ready to go
    recorder.start_recording(config['frame_width'])
    print("Ready:")

    while True:
        try:
            if recorder.has_new_data():
                # Get the most recent spoken data into the phrase buffer
                recorder.flush_queue_to_phrase_buffer()
                
                # Output the phrase buffer to a temporary file
                temp_file = recorder.output_phrase_buffer_to_file()

                # Get the transcription / translation
                result = translator.translate(temp_file)
                
                if args.verbose:
                    print(f"[[{[(s['text'], s['start'], s['end'], spoken) for s in result['segments'] if s['start'] > spoken - config['overlap_tolerance']]}]]")

                for s in result['segments']:
                    # If the segment is before where we've already spoken, don't speak
                    if s['start'] < max(spoken - config['overlap_tolerance'], 0.0):
                        continue
                    
                    # If the segment is first and no speech prob is high, don't speak
                    if s['start'] == 0.0 and s['no_speech_prob'] > config['no_speech_threshold']:
                        continue
                    
                    # If greedy policy, speak immediately
                    if config['policy'] == "greedy":
                        q.put(s['text'].strip())
                        spoken = s['end']
                        continue

                    # If confidence policy & confidence is high, speak
                    if config['policy'] == "confidence" and s['no_speech_prob'] < 1 - config['confidence_threshold']:
                        q.put(s['text'].strip())
                        spoken = s['end']
                        continue
                    
                    # If consensus policy, try to find consensus
                    if config['policy'] == "consensus" and prev:
                        # If the segment is past the end of where previous had, we're done
                        if s['id'] >= len(prev['segments']):
                            break
                        
                        # Check for consensus and if so, speak
                        if ratio(s['text'], prev['segments'][s['id']]['text']) >= config['consensus_threshold']:
                            q.put(s['text'].strip())
                            spoken = s['end']
                        else:
                            break

                # Save the current result to prev
                prev = result
                
                # This is the last time we received new audio data from the queue.
                phrase_time = datetime.utcnow()
                
            elif phrase_time and datetime.utcnow() - phrase_time > timedelta(seconds=config['phrase_timeout']):
                # if enough time has elapsed, speak all segments
                if prev:
                    for s in prev['segments']:
                        if s['start'] > spoken - config['overlap_tolerance']:
                            if s['start'] == 0.0 and s['no_speech_prob'] > 0.5:
                                continue
                            else:
                                q.put(s['text'].strip())
                prev = None
                spoken = 0.0
                recorder.clear_phrase_buffer()
                phrase_time = None
        except KeyboardInterrupt:
            break

def main():
    q = Queue()
    p = Process(target=recognize, args=(q,))
    p.daemon = True
    p.start()

    # Set up ElevenLabs request header
    url = keys['elevenlabs_api_url']
    headers = {
        "xi-api-key": keys['elevenlabs_api_key']
    }

    while True:
        try:
            data = q.get()

            if data is None: continue

            print(data)

            data = {
                "text": data,
                "optimize_streaming_latency": 3,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            response = requests.post(url, data=json.dumps(data), headers=headers)

            if response.status_code == 200:
                new_temp_file = NamedTemporaryFile().name + ".mp3"
                with open(new_temp_file, 'w+b') as f:
                    f.write(response.content)
                playsound(new_temp_file)
            else:
                print("Elevenlabs API Returned an Error")
        except KeyboardInterrupt:
            break
    
if __name__ == "__main__":
    main()