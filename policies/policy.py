from .baselines.baselines import GreedyPolicy, OfflinePolicy
from .confidence.confidence import ConfidenceAwarePolicy
from .consensus.consensus import ConsensusPolicy

class Policy:
    """Shared interface for all policies"""
    def apply(self, segment: dict) -> bool:
        """Apply the policy to a segment provided by the translator
           Return True if the segment should be spoken and False if it shouldn't"""
        pass
        
def get_policy(config: dict):
    if config['policy'] == "greedy":
        return GreedyPolicy()
    elif config['policy'] == "offline":
        return OfflinePolicy()
    elif config['policy'] == "confidence":
        return ConfidenceAwarePolicy(config['confidence_threshold'])
    elif config['policy'] == "consensus":
        return ConsensusPolicy(config['consensus_threshold'])
    else:
        print(f"Error: Policy '{config['policy']}' does not exist")
        return None