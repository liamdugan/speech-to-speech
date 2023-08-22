class ConfidenceAwarePolicy:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def apply(self, segment):
        return segment['no_speech_prob'] < 1 - self.alpha