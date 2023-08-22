import json, requests
from tempfile import NamedTemporaryFile
from playsound import playsound

class ElevenlabsVocalizer:
    def __init__(self, config, keys):
        # Set up ElevenLabs request header
        self.url = keys['elevenlabs_api_url']
        self.headers = {
            "xi-api-key": keys['elevenlabs_api_key']
        }

        self.stability = config['stability']
        self.similarity_boost = config['similarity_boost']
        self.optimize_streaming_latency = config['optimize_streaming_latency']
    
    def speak(self, text):
        data = {
            "text": text,
            "optimize_streaming_latency": self.optimize_streaming_latency,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost
            }
        }
        
        response = requests.post(self.url, data=json.dumps(data), headers=self.headers)

        if response.status_code == 200:
            new_temp_file = NamedTemporaryFile().name + ".mp3"
            with open(new_temp_file, 'w+b') as f:
                f.write(response.content)
            playsound(new_temp_file)
        else:
            print("Elevenlabs API Returned an Error")
