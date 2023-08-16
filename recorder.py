import speech_recognition as sr
from multiprocessing import Queue

class MicrophoneRecorder:
    def __init__(self, mic_id, energy_threshold, sampling_rate):
        self.source = sr.Microphone(sample_rate=sampling_rate, device_index=mic_id)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False
        
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.data_queue = Queue()

    def start_recording(self, frame_width):

        def record_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            self.data_queue.put(audio.get_raw_data())

        self.recorder.listen_in_background(self.source, 
                                           record_callback, 
                                           phrase_time_limit=frame_width)