from .whisper.whisper_api import WhisperAPITranslator
from .whisper.whisper_local import WhisperLocalTranslator

class Translator:
    """Shared interface for all translators"""
    def translate(self, file: str) -> dict:
        """Name of wav file containing speech goes in, JSON containing a list of segments comes out
           segment is a sequence of tokens with a single confidence score and no speech probability."""
        pass

def get_translator(name: str, use_local: bool, config: dict, keys: dict):
    if name == "whisper":
        if use_local:
            return WhisperLocalTranslator(config['model_name'], config['input_language'])
        else:
            return WhisperAPITranslator(keys["openai_api_key"], keys["openai_org_id"])
    else:
        print(f"Error: Translator '{name}' does not exist")
        return None