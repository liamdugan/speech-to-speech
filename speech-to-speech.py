import argparse, yaml
from multiprocessing import Process, Queue
from datetime import datetime, timedelta

from translators.translator import get_translator
from policies.policy import get_policy
from recorders.microphone_recorder import MicrophoneRecorder
from vocalizers.vocalizer import get_vocalizer

parser = argparse.ArgumentParser()
parser.add_argument("--microphone_id", default=2, help="ID for the input microphone", type=int)
parser.add_argument("--verbose", action='store_true', help='Whether to print out the intermediate results or not')
parser.add_argument("--use_local", action='store_true', help='Whether to use the local whisper model instead of the API')
parser.add_argument("--api_keys", default="api_keys.yml", help="The path to the api keys file", type=str)
parser.add_argument("--config", default="config.yml", help="The path to the config file", type=str)
args = parser.parse_args()

# Retrieve the API keys from the API keys file
with open(args.api_keys, "r") as f:
    keys = yaml.load(f, Loader=yaml.loader.SafeLoader)

# Retrieve the Config info from the config file
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    
def recognize(q):
    phrase_time = None
    spoken = 0.0
    prev = None

    # Initialize the Translator
    translator = get_translator("whisper", args.use_local, config, keys)
    
    # Initialize the Recorder
    recorder = MicrophoneRecorder(args.microphone_id, config['energy_threshold'], config['sampling_rate'])
    
    # Initialize the Policy
    policy = get_policy(config)
    
    # Start recording and cue the user that we're ready to go
    recorder.start_recording(config['frame_width'])
    print("Ready:")

    while True:
        if recorder.has_new_data():

            # Get the most recent spoken data into the phrase buffer
            recorder.flush_queue_to_phrase_buffer()
            
            # Output the phrase buffer to a temporary file
            temp_file = recorder.output_phrase_buffer_to_file()

            # Get the translation from the translator
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

                # If the policy returns that we should speak, speak
                if policy.apply(s):
                    q.put(s['text'].strip())
                    spoken = s['end']

            # Save the current result to prev
            prev = result
            policy.prev = result
            
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
            policy.prev = None
            spoken = 0.0
            recorder.clear_phrase_buffer()
            phrase_time = None

def main():
    q = Queue()
    p = Process(target=recognize, daemon=True, args=(q,))

    # Initialize vocalizer
    vocalizer = get_vocalizer("elevenlabs", config, keys)

    # Start the recognizer process
    p.start()

    while True:
        if data := q.get():
            print(data)
            vocalizer.speak(data)
    
if __name__ == "__main__":
    main()