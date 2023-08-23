import os

class Logger:
    @staticmethod
    def print_transcription(translation, result):
        spoken_text = ''.join(translation)
        hypothesis_text = ''.join([s['text'] for s in result['segments']])

        # Clear console and print spoken in cyan (\033[96m) and hypothesis in white (\033[00m)
        os.system('cls' if os.name=='nt' else 'clear')
        print(f"\033[96m{spoken_text}\033[00m{hypothesis_text}")
        print('', end='', flush=True)