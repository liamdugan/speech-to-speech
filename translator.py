import whisper
import openai

class Translator:
    """Shared interface for all translators"""
    def translate(self, file: str) -> list[dict]:
        """Filename containing speech goes in, list of chunks go out"""

class WhisperLocalTranslator(Translator):
    def __init__(self, model_name, input_lang):
        if model_name != "large" and input_lang.lower() == "english":
            model_name = model_name + ".en"
        self.model = whisper.load_model(model_name)
        self.fp16 = True
        self.input_lang = input_lang

    def translate(self, file):
        return self.model.transcribe(file, fp16=self.fp16, task="translate", language=self.input_lang)
    
class WhisperAPITranslator(Translator):
    def __init__(self, api_key, organization, input_lang):
        openai.api_key = api_key
        openai.organization = organization
        self.input_lang = input_lang
    
    def translate(self, file):
        audio_file = open(file, "rb")
        return openai.Audio.translate("whisper-1", audio_file, language="en", response_format="verbose_json")