from typing import List

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

from recipe.deep_grpo.protocol import Node, FINISH_REASON

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


@dataclass
class BranchingSelection:
    branch_chain_root: Node
    branch_chain_root_index: int
    total_length: int


@dataclass
class BranchingFeedback:
    total_length: int
    branch_chain_root_index: int
    is_success: bool
    reward: float = 0.0


class BranchingStrategy(ABC):    
    def __init__(self, max_model_len: int):
        self.max_model_len = max_model_len

    @abstractmethod
    def select_node(self, chain: List[Node], n_points: int, **kwargs) -> List[Node]:
        pass

    def update(self, feedback: BranchingFeedback):
        pass
    
    def _is_expandable(self, node: Node) -> bool:
        if node.finish_reason is None:
            return True # root node
        return (node.finish_reason == FINISH_REASON.STOP and 
                (len(node.prompt_ids) + len(node.response_ids) < self.max_model_len))
    

class RandomBranchingStrategy(BranchingStrategy):
    def select_node(self, chain: List[Node], n_points: int, **kwargs) -> List[Node]:
        expandable_candidates = [(i, node) for i, node in enumerate(chain) if self._is_expandable(node)]
        
        if not expandable_candidates:
            return []
        
        n_to_select = min(n_points, len(expandable_candidates))
        selected_candidates = random.sample(expandable_candidates, n_to_select)
        return [node for _, node in selected_candidates]


class UniformProbabilityModel:
    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        n_samples = len(X)
        return np.full((n_samples, 2), 1.0)

    @property
    def coef_(self):
        return [[0.0]]

    @property
    def intercept_(self):
        return [0.0]
    
    
class UtilitySamplingStrategy(BranchingStrategy):
    def __init__(self, max_model_len: int, prob_model_type="logistic", window_size=10000, position_bias=0.1, log_interval=1000):
        super().__init__(max_model_len)

        self.buffer_X = []
        self.buffer_y = []

        if prob_model_type == "logistic":
            self.model = LogisticRegression(solver='lbfgs')
        elif prob_model_type == "uniform":
            self.model = UniformProbabilityModel()
        else:
            raise ValueError(f"Unknown model_type: {prob_model_type}")

        self.window_size = window_size
        self.position_bias = position_bias

        logger.info(f"Utility SamplingStrategy initialized with model={prob_model_type}, window_size={window_size}, "
                    f"position_bias={position_bias}, log_interval={log_interval}")

        self.is_fitted = False

        self.update_count = 0
        self.LOG_INTERVAL = log_interval


    def select_node(self, chain: List[Node], n_points: int, **kwargs) -> List[Node]:
        candidates = [(i, node) for i, node in enumerate(chain) if self._is_expandable(node)]
        if not candidates: 
            return []
    
        total_length = len(chain)

        if len(self.buffer_y) > 10 and 0 in self.buffer_y and 1 in self.buffer_y:
            self.model.fit(self.buffer_X, self.buffer_y)
            self.is_fitted = True

        # if len(self.buffer_y) == self.window_size:
        #     breakpoint()

        indices = np.array([c[0] for c in candidates])
        ratios = (indices + 1) / total_length 

        if self.is_fitted:
            X_pred = ratios.reshape(-1, 1)
            prob_success = self.model.predict_proba(X_pred)[:, 1]

        else:
            prob_success = np.ones(len(candidates))

        pos_score = np.power(ratios, self.position_bias)

        weights = prob_success * pos_score
        assert np.sum(weights) > 1e-9
        p = weights / np.sum(weights)

        n_to_select = min(n_points, len(candidates))

        candidate_indices = np.random.choice(
            a=len(candidates), 
            size=n_to_select, 
            replace=False, 
            p=p
        )

        selected_nodes = [candidates[i][1] for i in candidate_indices]
    
        return selected_nodes
        

    def update(self, feedback: BranchingFeedback):
        ratio = (feedback.branch_chain_root_index + 1) / feedback.total_length
        assert ratio > 0 and ratio <= 1.0

        self.buffer_X.append([ratio])
        self.buffer_y.append(1 if feedback.is_success else 0)
        
        if len(self.buffer_X) > self.window_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        self.update_count += 1

        if self.update_count % self.LOG_INTERVAL == 0:
            self._log_model_status()

    def _log_model_status(self):
        logger.info(f"--- Logging model status at update #{self.update_count} ---")
        
        if self.is_fitted:
            w = self.model.coef_[0][0]
            b = self.model.intercept_[0]
            
            logger.info(f"Logistic Regression Parameters:")
            logger.info(f"  - Weights (W): {w:.6f}")
            logger.info(f"  - Bias (B):    {b:.6f}")    

        else:
            logger.info("Model is not fitted yet.")

        logger.info(f"Current buffer size: {len(self.buffer_X)}")
        logger.info(f"Success rate in buffer: {np.mean(self.buffer_y):.4f}")

        logger.info(f"--- End of log for update #{self.update_count} ---")

