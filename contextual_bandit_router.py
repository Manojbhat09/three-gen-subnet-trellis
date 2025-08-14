"""
Contextual Bandit LoRA Router with LinUCB and Thompson Sampling
Learns optimal LoRA selection from remote feedback in real-time
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import logging
from datetime import datetime
from collections import defaultdict, deque
import scipy.stats as stats
from dataclasses import dataclass
import threading
import time

@dataclass
class FeedbackRecord:
    """Record of LoRA performance feedback"""
    prompt: str
    lora_used: str
    remote_score: float
    local_score: float
    context_vector: np.ndarray
    timestamp: datetime
    validation_failed: bool = False

class PromptEmbedder:
    """Converts prompts to context vectors for bandit algorithms"""
    
    def __init__(self, max_features: int = 500):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Seed prompts for warm start
        self.seed_prompts = [
            "large dark purple pyramid shaped gemstone",
            "steel long-handled spade", 
            "robot that is orange and has pointed head",
            "cartoon character with big eyes",
            "geometric building structure",
            "colorful simple object",
            "realistic detailed weapon",
            "game icon symbol",
            "architectural clean design",
            "stylized bold character"
        ]
    
    def fit(self, prompts: List[str] = None):
        """Fit the embedder on prompts"""
        if prompts is None:
            prompts = self.seed_prompts
        
        # Fit TF-IDF
        tfidf_vectors = self.vectorizer.fit_transform(prompts)
        
        # Fit scaler
        self.scaler.fit(tfidf_vectors.toarray())
        self.is_fitted = True
        
        return self
    
    def transform(self, prompt: str) -> np.ndarray:
        """Convert prompt to context vector"""
        if not self.is_fitted:
            self.fit()
        
        tfidf_vector = self.vectorizer.transform([prompt])
        scaled_vector = self.scaler.transform(tfidf_vector.toarray())
        
        return scaled_vector[0]
    
    def get_dimension(self) -> int:
        """Get context vector dimension"""
        if not self.is_fitted:
            self.fit()
        return len(self.vectorizer.get_feature_names_out())

class ZeroScoreGateClassifier:
    """Prevents LoRAs that are likely to get 0 remote scores"""
    
    def __init__(self):
        self.classifiers = {}  # One classifier per LoRA
        self.min_samples = 10  # Minimum samples needed to train
        
        # Known risky combinations from data analysis
        self.hard_rules = {
            'baolei': ['tool', 'weapon', 'spade', 'crossbow'],
            'sd15_game_icon': ['robot', '3d', 'realistic'],
        }
    
    def _apply_hard_rules(self, lora: str, prompt: str) -> bool:
        """Apply hard-coded rules first"""
        prompt_lower = prompt.lower()
        if lora in self.hard_rules:
            risky_words = self.hard_rules[lora]
            if any(word in prompt_lower for word in risky_words):
                return False  # Likely to get 0 score
        return True
    
    def is_safe(self, lora: str, context_vector: np.ndarray, prompt: str = "") -> bool:
        """Check if LoRA is safe to use for this context"""
        # Apply hard rules first
        if not self._apply_hard_rules(lora, prompt):
            return False
        
        # Use learned classifier if available
        if lora in self.classifiers and len(self.classifiers[lora]['X']) >= self.min_samples:
            classifier = self.classifiers[lora]['model']
            if classifier is not None:
                prob_safe = classifier.predict_proba(context_vector.reshape(1, -1))[0][1]
                return prob_safe > 0.3  # Conservative threshold
        
        return True  # Default to safe if no data
    
    def update(self, lora: str, context_vector: np.ndarray, got_zero_score: bool):
        """Update classifier with new feedback"""
        if lora not in self.classifiers:
            self.classifiers[lora] = {
                'X': [],
                'y': [],
                'model': None
            }
        
        # Add new sample
        self.classifiers[lora]['X'].append(context_vector)
        self.classifiers[lora]['y'].append(0 if got_zero_score else 1)  # 0=unsafe, 1=safe
        
        # Retrain if we have enough samples
        if len(self.classifiers[lora]['X']) >= self.min_samples:
            X = np.array(self.classifiers[lora]['X'])
            y = np.array(self.classifiers[lora]['y'])
            
            # Only train if we have both classes
            if len(np.unique(y)) > 1:
                classifier = LogisticRegression(random_state=42)
                classifier.fit(X, y)
                self.classifiers[lora]['model'] = classifier

class LinUCBBandit:
    """LinUCB contextual bandit for LoRA selection"""
    
    def __init__(self, loras: List[str], context_dim: int, alpha: float = 1.0):
        self.loras = loras
        self.context_dim = context_dim
        self.alpha = alpha  # Exploration parameter
        
        # Initialize parameters for each LoRA
        self.A = {}  # Covariance matrices
        self.b = {}  # Reward vectors
        
        for lora in loras:
            self.A[lora] = np.eye(context_dim)
            self.b[lora] = np.zeros(context_dim)
    
    def select_lora(self, context: np.ndarray, safe_loras: List[str]) -> Tuple[str, float]:
        """Select LoRA using LinUCB algorithm"""
        ucb_values = {}
        
        for lora in safe_loras:
            if lora not in self.A:
                # Initialize new LoRA
                self.A[lora] = np.eye(self.context_dim)
                self.b[lora] = np.zeros(self.context_dim)
            
            # Compute UCB
            A_inv = np.linalg.inv(self.A[lora])
            theta = A_inv @ self.b[lora]
            
            # Expected reward
            expected_reward = context @ theta
            
            # Confidence bound
            confidence = self.alpha * np.sqrt(context @ A_inv @ context)
            
            # UCB value
            ucb_values[lora] = expected_reward + confidence
        
        # Select LoRA with highest UCB
        best_lora = max(ucb_values.keys(), key=lambda x: ucb_values[x])
        confidence = ucb_values[best_lora]
        
        return best_lora, confidence
    
    def update(self, lora: str, context: np.ndarray, reward: float):
        """Update bandit with observed reward"""
        if lora not in self.A:
            self.A[lora] = np.eye(self.context_dim)
            self.b[lora] = np.zeros(self.context_dim)
        
        # Update parameters
        self.A[lora] += np.outer(context, context)
        self.b[lora] += reward * context

class ThompsonSamplingBandit:
    """Thompson Sampling contextual bandit for LoRA selection"""
    
    def __init__(self, loras: List[str], context_dim: int, alpha: float = 1.0, beta: float = 1.0):
        self.loras = loras
        self.context_dim = context_dim
        self.alpha = alpha  # Prior precision
        self.beta = beta   # Noise precision
        
        # Initialize parameters for each LoRA
        self.mu = {}     # Mean parameters
        self.Sigma = {}  # Covariance matrices
        
        for lora in loras:
            self.mu[lora] = np.zeros(context_dim)
            self.Sigma[lora] = np.eye(context_dim) / alpha
    
    def select_lora(self, context: np.ndarray, safe_loras: List[str]) -> Tuple[str, float]:
        """Select LoRA using Thompson Sampling"""
        sampled_rewards = {}
        
        for lora in safe_loras:
            if lora not in self.mu:
                # Initialize new LoRA
                self.mu[lora] = np.zeros(self.context_dim)
                self.Sigma[lora] = np.eye(self.context_dim) / self.alpha
            
            # Sample theta from posterior
            theta_sample = np.random.multivariate_normal(self.mu[lora], self.Sigma[lora])
            
            # Compute expected reward for this sample
            sampled_rewards[lora] = context @ theta_sample
        
        # Select LoRA with highest sampled reward
        best_lora = max(sampled_rewards.keys(), key=lambda x: sampled_rewards[x])
        confidence = sampled_rewards[best_lora]
        
        return best_lora, confidence
    
    def update(self, lora: str, context: np.ndarray, reward: float):
        """Update bandit with observed reward"""
        if lora not in self.mu:
            self.mu[lora] = np.zeros(self.context_dim)
            self.Sigma[lora] = np.eye(self.context_dim) / self.alpha
        
        # Bayesian update
        Sigma_inv = np.linalg.inv(self.Sigma[lora])
        new_Sigma_inv = Sigma_inv + self.beta * np.outer(context, context)
        self.Sigma[lora] = np.linalg.inv(new_Sigma_inv)
        
        self.mu[lora] = self.Sigma[lora] @ (Sigma_inv @ self.mu[lora] + self.beta * reward * context)

class PrototypeKNNRouter:
    """Prototype + KNN routing for warm start"""
    
    def __init__(self):
        self.prototypes = {
            'cartoon_3d': [
                "large dark purple pyramid shaped gemstone",
                "colorful geometric shape",
                "simple cartoon object",
                "cute stylized character"
            ],
            'isometric_3d': [
                "clean architectural structure", 
                "geometric building block",
                "structured design element",
                "minimalist geometric form"
            ],
            'tf2_style': [
                "game character asset",
                "stylized weapon design", 
                "bold cartoon style",
                "team fortress character"
            ],
            'baolei': [
                "realistic detailed object",
                "photorealistic render",
                "high detail sculpture"
            ],
            'sd15_game_icon': [
                "simple game icon",
                "flat design symbol",
                "minimalist icon design"
            ]
        }
        
        self.embedder = PromptEmbedder()
        self.knn_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize KNN models for each LoRA"""
        all_prompts = []
        for prompts in self.prototypes.values():
            all_prompts.extend(prompts)
        
        self.embedder.fit(all_prompts)
        
        for lora, prompts in self.prototypes.items():
            if len(prompts) >= 2:
                vectors = np.array([self.embedder.transform(p) for p in prompts])
                knn = NearestNeighbors(n_neighbors=min(3, len(prompts)), metric='cosine')
                knn.fit(vectors)
                self.knn_models[lora] = knn
    
    def get_lora_scores(self, prompt: str) -> Dict[str, float]:
        """Get similarity scores for each LoRA"""
        scores = {}
        context_vector = self.embedder.transform(prompt)
        
        for lora, knn_model in self.knn_models.items():
            distances, _ = knn_model.kneighbors([context_vector])
            # Convert distance to similarity score
            avg_distance = np.mean(distances[0])
            similarity = np.exp(-avg_distance)  # Exponential decay
            scores[lora] = similarity
        
        return scores

class ContextualBanditLoRARouter:
    """Complete contextual bandit system for LoRA routing"""
    
    def __init__(self, 
                 loras: List[str] = None,
                 bandit_type: str = 'linucb',  # 'linucb' or 'thompson'
                 alpha: float = 1.0,
                 save_path: str = 'bandit_router_state.pkl'):
        
        if loras is None:
            loras = ['cartoon_3d', 'isometric_3d', 'tf2_style', 'baolei', 'sd15_game_icon']
        
        self.loras = loras
        self.bandit_type = bandit_type
        self.save_path = save_path
        
        # Initialize components
        self.embedder = PromptEmbedder()
        self.embedder.fit()  # Warm start with seed prompts
        
        self.gate_classifier = ZeroScoreGateClassifier()
        self.prototype_router = PrototypeKNNRouter()
        
        # Initialize bandit
        context_dim = self.embedder.get_dimension()
        if bandit_type == 'linucb':
            self.bandit = LinUCBBandit(loras, context_dim, alpha)
        else:
            self.bandit = ThompsonSamplingBandit(loras, context_dim, alpha)
        
        # Feedback storage
        self.feedback_history = deque(maxlen=10000)
        self.performance_stats = defaultdict(list)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Try to load existing state
        self.load_state()
    
    def select_lora(self, prompt: str, use_bandit: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Select best LoRA for prompt"""
        with self.lock:
            context_vector = self.embedder.transform(prompt)
            
            # Filter safe LoRAs
            safe_loras = [lora for lora in self.loras 
                         if self.gate_classifier.is_safe(lora, context_vector, prompt)]
            
            if not safe_loras:
                # Fallback to safest option
                safe_loras = ['cartoon_3d']
                self.logger.warning(f"No safe LoRAs found for prompt: {prompt[:50]}...")
            
            decision_info = {
                'prompt': prompt,
                'safe_loras': safe_loras,
                'filtered_loras': [l for l in self.loras if l not in safe_loras]
            }
            
            if use_bandit and len(self.feedback_history) > 10:
                # Use bandit for selection
                selected_lora, confidence = self.bandit.select_lora(context_vector, safe_loras)
                decision_info['method'] = 'bandit'
                decision_info['confidence'] = confidence
            else:
                # Use prototype + KNN for warm start
                prototype_scores = self.prototype_router.get_lora_scores(prompt)
                # Filter by safe LoRAs
                safe_scores = {lora: score for lora, score in prototype_scores.items() 
                              if lora in safe_loras}
                
                if safe_scores:
                    selected_lora = max(safe_scores.keys(), key=lambda x: safe_scores[x])
                    decision_info['confidence'] = safe_scores[selected_lora]
                else:
                    selected_lora = safe_loras[0]
                    decision_info['confidence'] = 0.5
                
                decision_info['method'] = 'prototype_knn'
                decision_info['prototype_scores'] = prototype_scores
            
            decision_info['selected_lora'] = selected_lora
            
            self.logger.info(f"🎯 Selected {selected_lora} for '{prompt[:30]}...' "
                           f"(method: {decision_info['method']}, confidence: {decision_info['confidence']:.3f})")
            
            return selected_lora, decision_info
    
    def update_feedback(self, 
                       prompt: str, 
                       lora_used: str, 
                       remote_score: float,
                       local_score: float = None,
                       validation_failed: bool = False):
        """Update router with feedback from remote validation"""
        with self.lock:
            context_vector = self.embedder.transform(prompt)
            
            # Create feedback record
            feedback = FeedbackRecord(
                prompt=prompt,
                lora_used=lora_used,
                remote_score=remote_score,
                local_score=local_score or 0.0,
                context_vector=context_vector,
                timestamp=datetime.now(),
                validation_failed=validation_failed
            )
            
            self.feedback_history.append(feedback)
            
            # Update bandit (use remote score as reward)
            reward = remote_score
            self.bandit.update(lora_used, context_vector, reward)
            
            # Update gate classifier
            got_zero_score = (remote_score == 0.0)
            self.gate_classifier.update(lora_used, context_vector, got_zero_score)
            
            # Update performance stats
            self.performance_stats[lora_used].append(remote_score)
            
            self.logger.info(f"📊 Updated feedback: {lora_used} got {remote_score:.3f} "
                           f"for '{prompt[:30]}...'")
            
            # Periodic save
            if len(self.feedback_history) % 50 == 0:
                self.save_state()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for lora in self.loras:
            scores = self.performance_stats[lora]
            if scores:
                stats[lora] = {
                    'count': len(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'zero_rate': sum(1 for s in scores if s == 0.0) / len(scores),
                    'recent_mean': np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
                }
            else:
                stats[lora] = {
                    'count': 0,
                    'mean_score': 0.0,
                    'std_score': 0.0,
                    'zero_rate': 0.0,
                    'recent_mean': 0.0
                }
        
        return stats
    
    def save_state(self):
        """Save router state to disk"""
        try:
            state = {
                'bandit': self.bandit,
                'gate_classifier': self.gate_classifier,
                'embedder': self.embedder,
                'feedback_history': list(self.feedback_history),
                'performance_stats': dict(self.performance_stats),
                'loras': self.loras,
                'bandit_type': self.bandit_type
            }
            
            with open(self.save_path, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"💾 Saved router state to {self.save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load router state from disk"""
        try:
            with open(self.save_path, 'rb') as f:
                state = pickle.load(f)
            
            self.bandit = state['bandit']
            self.gate_classifier = state['gate_classifier']
            self.embedder = state['embedder']
            self.feedback_history = deque(state['feedback_history'], maxlen=10000)
            self.performance_stats = defaultdict(list, state['performance_stats'])
            
            self.logger.info(f"📂 Loaded router state from {self.save_path}")
        except FileNotFoundError:
            self.logger.info("No existing state found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")

# Integration with existing orchestrator
class SmartContextualOrchestrator:
    """Integration with your existing orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.router = ContextualBanditLoRARouter(
            bandit_type='linucb',  # or 'thompson'
            alpha=1.5  # Higher = more exploration
        )
        self.logger = logging.getLogger(__name__)
    
    async def generate_3d_model(self, task) -> Optional[Dict[str, Any]]:
        """Generate 3D model with smart LoRA routing"""
        
        # Select LoRA using contextual bandit
        selected_lora, decision_info = self.router.select_lora(task.prompt)
        
        # Update endpoint
        endpoint = f"{self.config['generation_server_url']}/Image Generation/{selected_lora}/"
        
        self.logger.info(f"🤖 Contextual bandit selected: {selected_lora} "
                        f"(method: {decision_info['method']}, "
                        f"confidence: {decision_info['confidence']:.3f})")
        
        # Generate model (your existing logic)
        try:
            # ... your existing generation code ...
            result = await self._call_generation_endpoint(endpoint, task)
            
            # Update router with feedback
            if hasattr(task, 'task_fidelity_score'):
                self.router.update_feedback(
                    prompt=task.prompt,
                    lora_used=selected_lora,
                    remote_score=task.task_fidelity_score,
                    local_score=getattr(task, 'local_validation_score', None),
                    validation_failed=getattr(task, 'validation_failed', False)
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            # Still update with zero score for learning
            self.router.update_feedback(
                prompt=task.prompt,
                lora_used=selected_lora,
                remote_score=0.0,
                validation_failed=True
            )
            return None
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get router performance statistics"""
        return self.router.get_performance_stats()

# Usage example and testing
if __name__ == "__main__":
    # Initialize router
    router = ContextualBanditLoRARouter(bandit_type='linucb', alpha=1.5)
    
    # Test prompts from your data
    test_prompts = [
        "large dark purple pyramid shaped gemstone",
        "steel long-handled spade", 
        "robot that is orange and has pointed head",
        "cartoon character with big eyes",
        "geometric building structure",
        "realistic detailed weapon",
        "simple game icon design"
    ]
    
    print("🚀 Testing Contextual Bandit LoRA Router")
    print("=" * 50)
    
    # Simulate some interactions
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 Prompt {i+1}: '{prompt}'")
        
        # Select LoRA
        selected_lora, decision_info = router.select_lora(prompt)
        print(f"   🎯 Selected: {selected_lora}")
        print(f"   📊 Method: {decision_info['method']}")
        print(f"   🔒 Safe LoRAs: {decision_info['safe_loras']}")
        
        # Simulate feedback (you would get this from actual remote validation)
        if 'spade' in prompt and selected_lora == 'baolei':
            simulated_score = 0.0  # Known bad combination
        elif selected_lora == 'cartoon_3d':
            simulated_score = np.random.normal(0.7, 0.1)  # Generally good
        else:
            simulated_score = np.random.normal(0.5, 0.2)  # Average
        
        simulated_score = max(0.0, min(1.0, simulated_score))  # Clamp to [0,1]
        
        # Update with feedback
        router.update_feedback(prompt, selected_lora, simulated_score)
        print(f"   📈 Simulated score: {simulated_score:.3f}")
    
    # Show performance stats
    print("\n📊 Performance Statistics:")
    print("=" * 30)
    stats = router.get_performance_stats()
    for lora, lora_stats in stats.items():
        print(f"{lora}:")
        print(f"  Count: {lora_stats['count']}")
        print(f"  Mean score: {lora_stats['mean_score']:.3f}")
        print(f"  Zero rate: {lora_stats['zero_rate']:.1%}")
