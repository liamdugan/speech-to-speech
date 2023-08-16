from Levenshtein import ratio

class Policy:
    """Shared interface for all policies"""
    def apply(self, segment: dict) -> bool:
        """Apply the policy to a segment provided by the translator"""

    @staticmethod
    def get_policy(name: str, alpha: int, gamma: int):
        if name == "greedy":
            return GreedyPolicy()
        elif name == "offline":
            return OfflinePolicy()
        elif name == "confidence":
            return ConfidenceAwarePolicy(alpha)
        elif name == "consensus":
            return ConsensusPolicy(gamma)
        else:
            print(f"Error: Policy '{name}' does not exist")
            return None

class GreedyPolicy(Policy):
    def apply(self, _):
        return True

class OfflinePolicy(Policy):
    def apply(self, _):
        return False
    
class ConfidenceAwarePolicy(Policy):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def apply(self, segment):
        return segment['no_speech_prob'] < 1 - self.alpha
    
class ConsensusPolicy(Policy):
    def __init__(self, gamma):
        self.gamma = gamma
        self.prev = None

    def apply(self, segment):
        if not self.prev or segment['id'] >= len(self.prev['segments']):
            return False
        return ratio(segment['text'], self.prev['segments'][segment['id']]['text']) >= self.gamma