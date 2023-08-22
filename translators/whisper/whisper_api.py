import openai, os

class WhisperAPITranslator:
    def __init__(self, api_key, organization):
        openai.api_key = api_key
        openai.organization = organization
    
    def translate(self, file):
        audio_file = open(file, "rb")
        return openai.Audio.translate("whisper-1", audio_file, 
                                      language="en", 
                                      response_format="verbose_json")