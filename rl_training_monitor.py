#!/usr/bin/env python3
"""
RL Training Monitor & Production Readiness Assessor
==================================================
Monitors training progress and determines when model is ready for production.
Tracks convergence, performance metrics, and generalization ability.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from collections import deque
import statistics

@dataclass
class TrainingMetrics:
    """Comprehensive training metrics"""
    episode: int
    score: float
    reward: float
    epsilon: float
    loss: float
    ultra_achieved: bool
    improvement: float
    prompt_length: int
    action_type: str
    exploration_action: bool

@dataclass
class ConvergenceAnalysis:
    """Analysis of training convergence"""
    is_converged: bool
    convergence_episode: Optional[int]
    stability_score: float
    improvement_trend: float
    epsilon_stable: bool
    performance_stable: bool

class RLTrainingMonitor:
    """Monitors RL training progress and production readiness"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metrics_history = []
        self.convergence_window = deque(maxlen=window_size)
        
        # Success criteria
        self.success_criteria = {
            "ultra_achievement_rate": 0.3,      # 30%+ episodes achieve 0.96+
            "avg_score_threshold": 0.75,        # Average score 0.75+
            "improvement_rate": 0.7,            # 70%+ episodes show improvement
            "epsilon_stability": 0.05,          # Epsilon stable around 5%
            "score_stability": 0.05,            # Score variance < 0.05
            "min_episodes": 100                 # Minimum training episodes
        }
        
        print("📊 RL TRAINING MONITOR INITIALIZED")
        print(f"   📈 Window Size: {window_size}")
        print(f"   🎯 Success Criteria: {self.success_criteria}")

    def add_episode_metrics(self, metrics: TrainingMetrics):
        """Add episode metrics for monitoring"""
        
        self.metrics_history.append(metrics)
        self.convergence_window.append(metrics)
        
        # Real-time monitoring
        if len(self.metrics_history) % 10 == 0:
            self._print_progress_update()

    def _print_progress_update(self):
        """Print real-time progress update"""
        
        recent_metrics = list(self.convergence_window)[-10:]
        if not recent_metrics:
            return
        
        recent_scores = [m.score for m in recent_metrics]
        recent_ultras = [m.ultra_achieved for m in recent_metrics]
        recent_epsilon = recent_metrics[-1].epsilon
        
        avg_score = statistics.mean(recent_scores)
        ultra_rate = sum(recent_ultras) / len(recent_ultras)
        
        print(f"\n📊 PROGRESS UPDATE (Episode {len(self.metrics_history)})")
        print(f"   🏆 Recent Avg Score: {avg_score:.3f}")
        print(f"   🎉 Recent Ultra Rate: {ultra_rate:.1%}")
        print(f"   🎲 Current Epsilon: {recent_epsilon:.3f}")

    def analyze_convergence(self) -> ConvergenceAnalysis:
        """Analyze if training has converged"""
        
        if len(self.metrics_history) < 50:
            return ConvergenceAnalysis(
                is_converged=False,
                convergence_episode=None,
                stability_score=0.0,
                improvement_trend=0.0,
                epsilon_stable=False,
                performance_stable=False
            )
        
        recent_window = list(self.convergence_window)
        
        # Epsilon stability analysis
        recent_epsilons = [m.epsilon for m in recent_window[-20:]]
        epsilon_variance = statistics.variance(recent_epsilons) if len(recent_epsilons) > 1 else 1.0
        epsilon_stable = epsilon_variance < 0.001 and recent_epsilons[-1] < 0.1
        
        # Performance stability analysis
        recent_scores = [m.score for m in recent_window[-30:]]
        score_variance = statistics.variance(recent_scores) if len(recent_scores) > 1 else 1.0
        performance_stable = score_variance < self.success_criteria["score_stability"]
        
        # Improvement trend analysis
        if len(recent_scores) >= 20:
            early_avg = statistics.mean(recent_scores[:10])
            late_avg = statistics.mean(recent_scores[-10:])
            improvement_trend = late_avg - early_avg
        else:
            improvement_trend = 0.0
        
        # Overall stability score
        stability_score = 1.0 - min(epsilon_variance * 1000, 1.0) - min(score_variance * 20, 1.0)
        
        # Convergence decision
        is_converged = (
            epsilon_stable and 
            performance_stable and 
            len(self.metrics_history) >= self.success_criteria["min_episodes"] and
            improvement_trend > -0.05  # Not degrading
        )
        
        convergence_episode = len(self.metrics_history) if is_converged else None
        
        return ConvergenceAnalysis(
            is_converged=is_converged,
            convergence_episode=convergence_episode,
            stability_score=stability_score,
            improvement_trend=improvement_trend,
            epsilon_stable=epsilon_stable,
            performance_stable=performance_stable
        )

    def assess_production_readiness(self) -> Dict:
        """Comprehensive production readiness assessment"""
        
        if len(self.metrics_history) < 50:
            return {
                "ready_for_production": False,
                "reason": "Insufficient training data",
                "metrics": {},
                "recommendations": ["Continue training - need at least 50 episodes"]
            }
        
        recent_window = list(self.convergence_window)
        all_metrics = self.metrics_history
        
        # Calculate key metrics
        metrics = {}
        
        # Ultra achievement rate
        recent_ultras = [m.ultra_achieved for m in recent_window]
        metrics["ultra_achievement_rate"] = sum(recent_ultras) / len(recent_ultras)
        
        # Average score performance
        recent_scores = [m.score for m in recent_window]
        metrics["avg_score"] = statistics.mean(recent_scores)
        metrics["best_score"] = max(recent_scores)
        
        # Improvement rate (episodes that improved from baseline)
        baseline_scores = [m.score for m in all_metrics[:10]] if len(all_metrics) >= 10 else [0.5]
        baseline_avg = statistics.mean(baseline_scores)
        improvements = [m.score > baseline_avg for m in recent_window]
        metrics["improvement_rate"] = sum(improvements) / len(improvements)
        
        # Exploration/exploitation balance
        explorations = [m.exploration_action for m in recent_window]
        metrics["exploration_rate"] = sum(explorations) / len(explorations)
        
        # Convergence analysis
        convergence = self.analyze_convergence()
        metrics["converged"] = convergence.is_converged
        metrics["stability_score"] = convergence.stability_score
        
        # Check each criterion
        checks = {}
        checks["ultra_rate_ok"] = metrics["ultra_achievement_rate"] >= self.success_criteria["ultra_achievement_rate"]
        checks["avg_score_ok"] = metrics["avg_score"] >= self.success_criteria["avg_score_threshold"]
        checks["improvement_ok"] = metrics["improvement_rate"] >= self.success_criteria["improvement_rate"]
        checks["converged_ok"] = metrics["converged"]
        checks["min_episodes_ok"] = len(all_metrics) >= self.success_criteria["min_episodes"]
        
        # Overall readiness
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        readiness_score = passed_checks / total_checks
        
        ready_for_production = readiness_score >= 0.8  # 80% of checks must pass
        
        # Generate recommendations
        recommendations = []
        if not checks["ultra_rate_ok"]:
            recommendations.append(f"Ultra rate too low ({metrics['ultra_achievement_rate']:.1%} < {self.success_criteria['ultra_achievement_rate']:.1%})")
        if not checks["avg_score_ok"]:
            recommendations.append(f"Average score too low ({metrics['avg_score']:.3f} < {self.success_criteria['avg_score_threshold']:.3f})")
        if not checks["improvement_ok"]:
            recommendations.append(f"Improvement rate too low ({metrics['improvement_rate']:.1%} < {self.success_criteria['improvement_rate']:.1%})")
        if not checks["converged_ok"]:
            recommendations.append("Training not converged - continue training")
        if not checks["min_episodes_ok"]:
            recommendations.append(f"Need more episodes ({len(all_metrics)} < {self.success_criteria['min_episodes']})")
        
        if ready_for_production:
            recommendations = ["✅ READY FOR PRODUCTION DEPLOYMENT!"]
        
        return {
            "ready_for_production": ready_for_production,
            "readiness_score": readiness_score,
            "metrics": metrics,
            "checks": checks,
            "convergence": convergence,
            "recommendations": recommendations,
            "total_episodes": len(all_metrics)
        }

    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        
        assessment = self.assess_production_readiness()
        
        report = f"""
🎓 RL TRAINING REPORT
{'='*60}

📊 TRAINING STATISTICS:
   Total Episodes: {assessment['total_episodes']}
   Ultra Achievement Rate: {assessment['metrics']['ultra_achievement_rate']:.1%}
   Average Score: {assessment['metrics']['avg_score']:.3f}
   Best Score: {assessment['metrics']['best_score']:.3f}
   Improvement Rate: {assessment['metrics']['improvement_rate']:.1%}
   Current Exploration: {assessment['metrics']['exploration_rate']:.1%}

🎯 PRODUCTION READINESS:
   Overall Score: {assessment['readiness_score']:.1%}
   Status: {"✅ READY" if assessment['ready_for_production'] else "❌ NOT READY"}

📈 CONVERGENCE ANALYSIS:
   Converged: {"✅ Yes" if assessment['convergence'].is_converged else "❌ No"}
   Stability Score: {assessment['convergence'].stability_score:.3f}
   Improvement Trend: {assessment['convergence'].improvement_trend:+.3f}

✅ READINESS CHECKS:
"""
        
        for check, passed in assessment['checks'].items():
            status = "✅" if passed else "❌"
            report += f"   {status} {check.replace('_', ' ').title()}\n"
        
        report += f"\n💡 RECOMMENDATIONS:\n"
        for rec in assessment['recommendations']:
            report += f"   • {rec}\n"
        
        return report

    def save_training_plots(self, output_dir: str = "training_plots"):
        """Generate and save training visualization plots"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if len(self.metrics_history) < 10:
            print("Not enough data for plots")
            return
        
        episodes = [m.episode for m in self.metrics_history]
        scores = [m.score for m in self.metrics_history]
        rewards = [m.reward for m in self.metrics_history]
        epsilons = [m.epsilon for m in self.metrics_history]
        ultras = [m.ultra_achieved for m in self.metrics_history]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score progression
        ax1.plot(episodes, scores, 'b-', alpha=0.7, label='Episode Score')
        ax1.axhline(y=0.96, color='r', linestyle='--', label='Ultra Target')
        ax1.axhline(y=0.8, color='orange', linestyle='--', label='Good Target')
        ax1.set_title('Score Progression')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Validation Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax2.plot(episodes, epsilons, 'g-', label='Epsilon')
        ax2.set_title('Exploration vs Exploitation')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon (Exploration Rate)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Reward progression
        ax3.plot(episodes, rewards, 'purple', alpha=0.7, label='Episode Reward')
        ax3.set_title('Reward Progression')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Total Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Ultra achievement rate (rolling average)
        window = 20
        if len(ultras) >= window:
            ultra_rolling = []
            for i in range(window-1, len(ultras)):
                window_ultras = ultras[i-window+1:i+1]
                ultra_rolling.append(sum(window_ultras) / len(window_ultras))
            
            ax4.plot(episodes[window-1:], ultra_rolling, 'red', label=f'Ultra Rate ({window}-ep window)')
            ax4.axhline(y=0.3, color='orange', linestyle='--', label='Target Rate (30%)')
            ax4.set_title('Ultra Achievement Rate')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Ultra Achievement Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved to {output_path}")

def create_training_monitor_integration():
    """Show how to integrate monitor with RL training"""
    
    integration_code = '''
# Integration with RL Training:

from rl_training_monitor import RLTrainingMonitor, TrainingMetrics

class MonitoredRLOptimizer(FixedRLOptimizer):
    def __init__(self, ultra_target: float = 0.96):
        super().__init__(ultra_target)
        self.monitor = RLTrainingMonitor()
        
    def train_episode(self, target_prompt: str, episode_num: int) -> Dict:
        result = super().train_episode(target_prompt, episode_num)
        
        # Add metrics to monitor
        metrics = TrainingMetrics(
            episode=episode_num,
            score=result['best_score'],
            reward=result['total_reward'],
            epsilon=result['epsilon'],
            loss=result.get('loss', 0.0),
            ultra_achieved=result['ultra_achieved'],
            improvement=result.get('improvement', 0.0),
            prompt_length=len(result.get('final_prompt', '')),
            action_type=result.get('final_action', 'unknown'),
            exploration_action=result['epsilon'] > 0.1
        )
        
        self.monitor.add_episode_metrics(metrics)
        
        # Check if ready for production every 25 episodes
        if episode_num % 25 == 0:
            assessment = self.monitor.assess_production_readiness()
            print(f"\\n📊 PRODUCTION READINESS CHECK:")
            print(f"   Status: {'✅ READY' if assessment['ready_for_production'] else '❌ NOT READY'}")
            print(f"   Score: {assessment['readiness_score']:.1%}")
            
            if assessment['ready_for_production']:
                print("\\n🎉 MODEL IS READY FOR PRODUCTION!")
                self.monitor.save_training_plots()
                return result
        
        return result
'''
    
    print("🔧 INTEGRATION CODE:")
    print(integration_code)

def main():
    """Demo the training monitor"""
    
    print("📊 RL TRAINING MONITOR DEMO")
    print("="*50)
    
    monitor = RLTrainingMonitor()
    
    # Simulate training progress
    print("\n🎮 Simulating training episodes...")
    
    for episode in range(1, 101):
        # Simulate realistic training progression
        base_score = 0.4 + (episode / 200)  # Gradual improvement
        noise = np.random.normal(0, 0.1)
        score = max(0, min(1, base_score + noise))
        
        epsilon = max(0.05, 0.9 * (0.99 ** episode))  # Epsilon decay
        ultra_achieved = score >= 0.96
        reward = (score - 0.5) * 100 + np.random.normal(0, 10)
        
        metrics = TrainingMetrics(
            episode=episode,
            score=score,
            reward=reward,
            epsilon=epsilon,
            loss=abs(np.random.normal(0.5, 0.2)),
            ultra_achieved=ultra_achieved,
            improvement=score - 0.5,
            prompt_length=np.random.randint(80, 120),
            action_type="APPLY_PATTERN",
            exploration_action=np.random.random() < epsilon
        )
        
        monitor.add_episode_metrics(metrics)
    
    # Generate final assessment
    print("\n" + monitor.generate_training_report())
    
    # Show integration example
    create_training_monitor_integration()

if __name__ == "__main__":
    main() 