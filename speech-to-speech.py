from multiprocessing import Process, Queue
import requests, wave, json, argparse, whisper, torch, io, os, queue
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
from time import sleep
from fuzzywuzzy import process
import pyaudio
from playsound import playsound
import sounddevice as sd
from pydub import AudioSegment

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="medium", help="Model to use",
                    choices=["tiny", "base", "small", "medium", "large"])
parser.add_argument("--energy_threshold", default=600,
                    help="Energy level for mic to detect.", type=int)
parser.add_argument("--record_timeout", default=3,
                    help="How real time the recording is in seconds.", type=float)
parser.add_argument("--phrase_timeout", default=15,
                    help="How much empty space between recordings before we "
                            "consider it a new line in the transcription.", type=float)  
parser.add_argument("--microphone_id", default=1,
                    help="ID for the input microphone", type=int)
args = parser.parse_args()

def find_longest_common_substring(s1, s2):
    match = process.extractBests(s1, [s2[:i] for i in range(len(s2))], limit=None)
    sorted_matches = sorted(match, key=lambda x: (-x[1], -len(x[0])))
    return sorted_matches[0][0]

#Eleven Labs
url = "https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX"
headers = {
    "xi-api-key": "fff703947a1ec48a8e57abc4507bcb30"
}

source = sr.Microphone(sample_rate=16000, device_index=args.microphone_id)

def record(q):
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    with source: 
        recorder.adjust_for_ambient_noise(source)

        while True:
            try:
                audio = recorder.listen(source, phrase_time_limit=args.record_timeout)
                q.put(audio.get_raw_data())
            except KeyboardInterrupt:
                q.put(None)
                break


def recognize(q):
    phrase_time = None
    current_phrase = ''
    phrase_buffer = bytes()
    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    # Load / Download model
    model = args.model
    audio_model = whisper.load_model(model)

    audio_queue = Queue()

    p = Process(target=record, args=(audio_queue, ))
    p.start()

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not audio_queue.empty():
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    phrase_buffer = bytes()
                    current_phrase = ''
                
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                while not audio_queue.empty():
                    data = audio_queue.get()
                    phrase_buffer += data

                audio_data = sr.AudioData(phrase_buffer, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), task="translate", language="japanese")
                text = result['text'].strip()

                # If this is a new phrase then we just put the text into the queue
                if not current_phrase:
                    q.put(text)
                else:
                    # If not, calculate the overlap between tokens of the phrase and the new text
                    substring = find_longest_common_substring(current_phrase, text)
                    q.put(text[len(substring):])

                current_phrase = text

        except KeyboardInterrupt:
            q.put(None)
            break

if __name__ == '__main__':
    q = Queue()
    p = Process(target=recognize, args=(q,))
    p.start()

    while True:
        try:
            data = q.get()

            if data is None: break

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
                print("Error")
        except KeyboardInterrupt:
            p.join()
            break