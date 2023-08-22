from Levenshtein import ratio

class ConsensusPolicy:
    def __init__(self, gamma):
        self.gamma = gamma
        self.prev = None

    def apply(self, segment):
        if not self.prev or segment['id'] >= len(self.prev['segments']):
            return False
        return ratio(segment['text'], self.prev['segments'][segment['id']]['text']) >= self.gamma