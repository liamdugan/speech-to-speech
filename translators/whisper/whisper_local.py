import whisper

class WhisperLocalTranslator:
    def __init__(self, model_name, input_lang):
        self.model = whisper.load_model(model_name)
        self.input_lang = input_lang

    def change_input_language(self, language):
        self.input_lang = language

    def translate(self, file, prompt):
        return self.model.transcribe(file, fp16=True, 
                                     task="translate", prompt=prompt,
                                     language=self.input_lang)