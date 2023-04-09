from multiprocessing import Process, Queue
import requests, wave, json, argparse, whisper, torch, io, os, queue
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
from time import sleep
import pyaudio
from playsound import playsound

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="medium", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
parser.add_argument("--energy_threshold", default=600, help="Energy level for mic to detect.", type=int)
parser.add_argument("--record_timeout", default=0.5, help="How often we run audio through whisper", type=float)
parser.add_argument("--phrase_timeout", default=1, help="How much silence before we clear the buffer.", type=float)  
parser.add_argument("--microphone_id", default=1, help="ID for the input microphone", type=int)
parser.add_argument("--verbose", action='store_true', help='Whether to print out the intermediate results or not')
args = parser.parse_args()

#Eleven Labs
url = "https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX"
headers = {
    "xi-api-key": "fff703947a1ec48a8e57abc4507bcb30"
}
    
def recognize(q):
    phrase_time = None
    spoken = 0.0
    tolerance = 1.0
    prev = None
    phrase_buffer = bytes()
    temp_file = NamedTemporaryFile().name
    
    # Load / Download model
    model = args.model
    audio_model = whisper.load_model(model)
    
    source = sr.Microphone(sample_rate=16000, device_index=args.microphone_id)

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    data_queue = Queue()

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

                while not data_queue.empty():
                    data = data_queue.get()
                    phrase_buffer += data

                audio_data = sr.AudioData(phrase_buffer, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), task="translate", language="japanese")

                if args.verbose:
                    print(f"[[{[s['text'] for s in result['segments'] if s['start'] > spoken - tolerance]}]]")

                # If there was a previous result, try to find consensus
                if prev:
                    for s in result['segments']:
                        # If the segment is before where we've already spoken, don't bother
                        if s['start'] <= spoken - tolerance:
                            continue
                        
                        # If the segment is first and no speech is high, don't speak it
                        if s['start'] == 0.0 and s['no_speech_prob'] > 0.7:
                            continue
                        
                        # If the segment is past the prev we're done
                        if s['id'] >= len(prev['segments']):
                            break
                        
                        # Check if there is consensus and if so, speak
                        if s['text'] == prev['segments'][s['id']]['text']:
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
                        if s['start'] > spoken - tolerance:
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
                print("Error")
        except KeyboardInterrupt:
            break
    
if __name__ == "__main__":
    main()