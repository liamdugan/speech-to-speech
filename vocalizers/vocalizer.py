from .elevenlabs.elevenlabs_vocalizer import ElevenlabsVocalizer

class Vocalizer:
    """Shared interface for all vocalizers"""
    def speak(self, text: str) -> None:
        """Text to speak comes in, speech happens (no return value)"""
        pass

def get_vocalizer(name: str, config: dict, keys: dict):
    if name == "elevenlabs":
        return ElevenlabsVocalizer(config, keys)
    else:
        print(f"Error: Vocalizer '{name}' does not exist")
        return None