import argparse, yaml
from multiprocessing import Process, Queue

from translators.translator import get_translator
from policies.policy import get_policy
from recorders.microphone_recorder import MicrophoneRecorder
from vocalizers.vocalizer import get_vocalizer
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--mic", default=0, help="ID for the input microphone", type=int)
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
    # time that we have spoken until at this point
    spoken = 0.0
    translation = []

    # Initialize the Translator and Policy
    translator = get_translator("whisper", args.use_local, config, keys)
    policy = get_policy(config)
    
    # Start recording and cue the user that we're ready to go
    recorder = MicrophoneRecorder(args.mic, config)
    recorder.start_recording()

    print("Ready:")
    while True:
        # Get the most recent spoken data into the phrase buffer
        recorder.flush_queue_to_phrase_buffer()

        if recorder.has_new_data():
            # Output the phrase buffer to a temporary file
            temp_file = recorder.output_phrase_buffer_to_file()

            # Get the translation from the translator
            result = translator.translate(temp_file, prompt=''.join(translation))

            for s in result['segments']:
                # If no speech prob is high, don't speak
                if s['no_speech_prob'] > config['no_speech_threshold']:
                    continue

                # If the policy returns that we should speak, speak
                if policy.apply(s):
                    q.put(s['text'].strip())
                    translation.append(s['text'])
                    spoken = s['end']

            # Save all segments we haven't yet spoken to prev
            result['segments'] = [s for s in result['segments'] if s['end'] > spoken]
            policy.prev = result

            # Log the unspoken hypotheses and spoken text to the console
            Logger.print_transcription(translation, result, args.verbose)

            # Clear all spoken audio from the phrase buffer and reset spoken pointer
            recorder.trim_phrase_buffer(spoken)
            spoken = 0.0

def main():
    q = Queue()
    p = Process(target=recognize, daemon=True, args=(q,))

    # Initialize vocalizer
    vocalizer = get_vocalizer("elevenlabs", config, keys)

    # Start the recognizer process
    p.start()

    # Loop infinitely and speak whenever we get new data
    while True:
        if data := q.get():
            vocalizer.speak(data)
    
if __name__ == "__main__":
    main()