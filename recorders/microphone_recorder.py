import speech_recognition as sr
from multiprocessing import Queue
import io
from tempfile import NamedTemporaryFile

class MicrophoneRecorder:
    def __init__(self, mic_id, energy_threshold, sampling_rate):
        # Initialize the microphone and the noise thresholding recognizer
        self.source = sr.Microphone(sample_rate=sampling_rate, device_index=mic_id)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False
        
        # Automatically adjust for ambient noise
        with self.source:        
            self.recorder.adjust_for_ambient_noise(self.source)

        # Data queue for communication with the background recording thread
        self.data_queue = Queue()
        # Buffer to keep track of all unspoken audio thus far
        self.phrase_buffer = bytes()
        # Temporary file to write to for input to whisper
        self.temp_file = NamedTemporaryFile().name + ".wav"

    def start_recording(self, frame_width):

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            self.data_queue.put(audio.get_raw_data())

        self.recorder.listen_in_background(self.source, 
                                           record_callback, 
                                           phrase_time_limit=frame_width)

    def has_new_data(self):
        return not self.data_queue.empty()

    def clear_phrase_buffer(self):
        self.phrase_buffer = bytes()
        
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