from multiprocessing import Process, Queue
import requests, wave, json, argparse, torch, io, os, queue, yaml
import whisper
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
from time import sleep
import pyaudio
from playsound import playsound
import time
from Levenshtein import ratio

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="small", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
parser.add_argument("--energy_threshold", default=600, help="Energy level for mic to detect.", type=int)
parser.add_argument("--tolerance", default=1.0, help="The tolerance for considering segments as passed", type=float)
parser.add_argument("--record_timeout", default=1.0, help="How often we run audio through whisper", type=float)
parser.add_argument("--phrase_timeout", default=2.0, help="How much silence before we clear the buffer.", type=float)
parser.add_argument("--consensus_threshold", default=0.75, help="If edit distance ratio > threshold then we speak the segment", type=float)
parser.add_argument("--input_language", default="English", help="The language that will be spoken as input", type=str)
parser.add_argument("--translate", action='store_true', help="Whether or not to translate the input (if not included text will be transcribed)")
parser.add_argument("--no_consensus", action='store_true', help="Disable checking for consensus")
parser.add_argument("--microphone_id", default=1, help="ID for the input microphone", type=int)
parser.add_argument("--verbose", action='store_true', help='Whether to print out the intermediate results or not')
parser.add_argument("--api_keys_path", default="api_keys.yml", help="The path to the api keys file", type=str)
args = parser.parse_args()

with open(args.api_keys_path, "r") as f:
    try:
        keys = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# ElevenLabs
url = keys['elevenlabs_api_url']
headers = {
    "xi-api-key": keys['elevenlabs_api_key']
}
    
def recognize(q):
    phrase_time = None
    spoken = 0.0
    prev = None
    phrase_buffer = bytes()
    temp_file = NamedTemporaryFile().name
    
    # Load / Download model
    model = args.model
    if args.model != "large" and args.input_language.lower() == "english":
        model = model + ".en"
    audio_model = whisper.load_model(model)
    
    source = sr.Microphone(sample_rate=16000, device_index=args.microphone_id)

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    data_queue = Queue()
    
    task = "translate" if args.translate else "transcribe"

    with source: 
        recorder.adjust_for_ambient_noise(source)
        
    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    # Cue the user that we're ready to go.
    print("Ready:")
    while True:
        try:
            if not data_queue.empty():

                # Get all data from the buffer
                while not data_queue.empty():
                    data = data_queue.get()
                    phrase_buffer += data

                audio_data = sr.AudioData(phrase_buffer, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Get the transcription / translation
                time1 = time.time()
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), task=task, language=args.input_language)
                time2 = time.time()
                
                if args.verbose:
                    print(f"({time2 - time1}) [[{[(s['text'], s['start'], s['end']) for s in result['segments'] if s['start'] > spoken - args.tolerance]}]]")

                for s in result['segments']:
                    # If the segment is before where we've already spoken, don't bother
                    if s['start'] < max(spoken - args.tolerance, 0.0):
                        continue
                    
                    # If the segment is first and no speech prob is high, don't speak
                    if s['start'] == 0.0 and s['no_speech_prob'] > 0.7:
                        continue
                    
                    # If we are not using consensus just speak whatever we have immediately
                    if args.no_consensus:
                        q.put(s['text'].strip())
                        spoken = s['end']
                        continue
                    
                    # If there was a previous result, try to find consensus
                    if prev:
                        # If the segment is past the end of where previous had, we're done
                        if s['id'] >= len(prev['segments']):
                            break
                        
                        # Check for consensus and if so, speak
                        if ratio(s['text'], prev['segments'][s['id']]['text']) > args.consensus_threshold:
                            q.put(s['text'].strip())
                            spoken = s['end']
                        else:
                            break

                # Save the current result to prev
                prev = result
                
                # This is the last time we received new audio data from the queue.
                phrase_time = datetime.utcnow()
                
            elif phrase_time and datetime.utcnow() - phrase_time > timedelta(seconds=args.phrase_timeout):
                # if enough time has elapsed, speak all segments currently in the buffer
                if prev:
                    for s in prev['segments']:
                        if s['start'] > spoken - args.tolerance:
                            if s['start'] == 0.0 and s['no_speech_prob'] > 0.5:
                                continue
                            else:
                                q.put(s['text'].strip())
                prev = None
                spoken = 0.0
                phrase_buffer = bytes()
                phrase_time = None
        except KeyboardInterrupt:
            break

def main():
    q = Queue()
    p = Process(target=recognize, args=(q,))
    p.daemon = True
    p.start()

    while True:
        try:
            data = q.get()

            if data is None: continue

            print(data)

            data = {
                "text": data,
                "voice_settings": {
                    "stability": 0,
                    "similarity_boost": 0
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