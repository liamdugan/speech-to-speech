import speech_recognition as sr
from multiprocessing import Queue
import io
from tempfile import NamedTemporaryFile

class MicrophoneRecorder:
    def __init__(self, mic_id, config):
        # Initialize the microphone and the noise thresholding recognizer
        self.source = sr.Microphone(sample_rate=config['sampling_rate'], device_index=mic_id)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = config['energy_threshold']
        self.recorder.dynamic_energy_threshold = False

        # Minimum amount of time before we send audio to the translator
        self.frame_width = config['frame_width']
        
        # Automatically adjust for ambient noise
        with self.source:        
            self.recorder.adjust_for_ambient_noise(self.source)

        # Data queue for communication with the background recording thread
        self.data_queue = Queue()
        # Buffer to keep track of all unspoken audio thus far
        self.phrase_buffer = bytes()
        # Temporary file to write to for input to whisper
        self.temp_file = NamedTemporaryFile().name + ".wav"

    def start_recording(self):

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            self.data_queue.put(audio.get_raw_data())

        self.recorder.listen_in_background(self.source, 
                                           record_callback, 
                                           phrase_time_limit=self.frame_width)

    def has_new_data(self):
        # Return true if there is at least frame_width seconds of audio in the phrase buffer
        min_frames = self.frame_width * self.source.SAMPLE_RATE * self.source.SAMPLE_WIDTH
        return len(self.phrase_buffer) > min_frames
    
    def clear_phrase_buffer(self):
        self.phrase_buffer = bytes()

    def trim_phrase_buffer(self, pointer):
        # edit phrase buffer to get rid of all speech before the pointer
        byte_index = int(pointer*self.source.SAMPLE_RATE*self.source.SAMPLE_WIDTH)
        self.phrase_buffer = self.phrase_buffer[byte_index:]
        
    def flush_queue_to_phrase_buffer(self):
        while not self.data_queue.empty():
            data = self.data_queue.get()
            self.phrase_buffer += data
        
    def output_phrase_buffer_to_file(self):
        # Convert data in phrase buffer to wav data
        audio_data = sr.AudioData(self.phrase_buffer, 
                                  self.source.SAMPLE_RATE, 
                                  self.source.SAMPLE_WIDTH)
        wav_data = io.BytesIO(audio_data.get_wav_data())

        # Write wav data to the file as bytes.
        with open(self.temp_file, 'w+b') as f:
            f.write(wav_data.read())

        return self.temp_file