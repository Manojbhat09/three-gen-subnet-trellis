#!/usr/bin/env python3
"""
Adaptive Learning Prompt Optimizer v2.0 with Multi-Strategy Testing
Purpose: Use AI reasoning to continuously learn and improve prompt optimization strategies
with multiple strategy attempts, continuous scoring, and learned feedback integration.
"""
import requests
import json
import time
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import statistics

@dataclass
class OptimizationAttempt:
    """Single strategy attempt within an optimization session"""
    strategy_name: str
    optimized_prompt: str
    validation_score: float
    demo_fidelity_score: float
    score_improvement: float
    fidelity_improvement: float
    attempt_number: int
    timestamp: float

@dataclass
class OptimizationSession:
    """Complete optimization session for one prompt with multiple attempts"""
    original_prompt: str
    prompt_category: str
    baseline_score: float
    baseline_fidelity: float
    attempts: List[OptimizationAttempt]
    best_attempt: Optional[OptimizationAttempt]
    total_attempts: int
    session_improvement: float
    session_success: bool
    timestamp: float

class AdaptiveLearningOptimizerV2:
    """Enhanced AI-powered adaptive prompt optimizer with multi-strategy testing"""
    
    def __init__(self, max_attempts_per_prompt: int = 5, min_improvement_threshold: float = 0.05):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "adaptive_learning_v2.db"
        self.max_attempts = max_attempts_per_prompt
        self.min_improvement_threshold = min_improvement_threshold
        self.optimization_sessions: List[OptimizationSession] = []
        
        # Enhanced strategies with learned insights
        self.base_strategies = {
            "raw": "{prompt}",
            "material_focus": "wbgmsst, solid {prompt} object 3D, white background",
            "geometric_focus": "wbgmsst, {prompt} geometric 3D model, white background", 
            "basic_description": "3D model of {prompt}",
            "current_production": "wbgmsst, {prompt} 3D isometric accurate, white background",
            "enhanced_clarity": "wbgmsst, detailed 3D {prompt} model, accurate geometry, white background",
            "concrete_object": "wbgmsst, {prompt} as 3D object, realistic proportions, white background",
            "minimal_enhancement": "{prompt}, 3D object",
            "simplified_description": "simple 3D {prompt}",
            "artistic_focus": "wbgmsst, artistic {prompt} sculpture, clean design, white background"
        }
        
        # Learned insights from previous feedback
        self.learned_patterns = {
            "high_baseline_avoid": ["material_focus", "enhanced_clarity"],  # Avoid for prompts scoring >0.8
            "decorative_prefer": ["raw", "basic_description"],
            "technical_prefer": ["material_focus", "minimal_enhancement"],
            "physical_mixed": ["material_focus", "raw", "concrete_object"],
            "abstract_prefer": ["geometric_focus", "artistic_focus"],
            "fashion_prefer": ["basic_description", "minimal_enhancement"]
        }
        
        # Strategy performance tracking
        self.strategy_performance = {}
        
        self.setup_database()
        self.load_historical_performance()
        
    def setup_database(self):
        """Initialize enhanced SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_prompt TEXT,
                prompt_category TEXT,
                baseline_score REAL,
                baseline_fidelity REAL,
                total_attempts INTEGER,
                best_strategy TEXT,
                best_score REAL,
                best_fidelity REAL,
                session_improvement REAL,
                session_success BOOLEAN,
                timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                strategy_name TEXT,
                optimized_prompt TEXT,
                validation_score REAL,
                demo_fidelity_score REAL,
                score_improvement REAL,
                fidelity_improvement REAL,
                attempt_number INTEGER,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES optimization_sessions (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                category TEXT,
                avg_score_improvement REAL,
                avg_fidelity_improvement REAL,
                success_rate REAL,
                usage_count INTEGER,
                last_updated REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_historical_performance(self):
        """Load historical strategy performance from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT strategy_name, category, avg_score_improvement, avg_fidelity_improvement, 
                   success_rate, usage_count 
            FROM strategy_performance_log
        ''')
        
        for row in cursor.fetchall():
            strategy, category, score_imp, fidelity_imp, success_rate, usage = row
            
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {}
            
            self.strategy_performance[strategy][category] = {
                'avg_score_improvement': score_imp,
                'avg_fidelity_improvement': fidelity_imp,
                'success_rate': success_rate,
                'usage_count': usage
            }
        
        conn.close()
        
    def categorize_prompt(self, prompt: str) -> str:
        """Use DeepSeek-R1 to categorize the prompt type with learned patterns"""
        
        system_prompt = """You are an expert at categorizing prompts for 3D model generation optimization.

Based on discovered patterns:
- physical_object: Concrete physical items (bucket, chair, bottle, etc.)
- technical_description: Technical/geometric descriptions (cylindrical, angular, measurements)
- abstract_artistic: Artistic or abstract concepts (sleek, reflecting, ethereal)
- fashion_clothing: Clothing and fashion items (shirt, dress, shoes)
- decorative_standard: Decorative objects or well-described standard items

Previous learning shows:
- Decorative items often work best with minimal changes
- Technical descriptions benefit from material focus
- Physical objects have mixed results and need careful strategy selection

Respond with ONLY the category name."""

        user_prompt = f"Categorize this prompt: '{prompt}'"
        
        try:
            response = self.query_deepseek(system_prompt, user_prompt)
            category = response.strip().lower()
            
            valid_categories = ["physical_object", "technical_description", "abstract_artistic", 
                             "fashion_clothing", "decorative_standard"]
            
            return category if category in valid_categories else self.heuristic_categorize(prompt)
                
        except Exception as e:
            print(f"🤖 AI categorization failed: {e}")
            return self.heuristic_categorize(prompt)
    
    def heuristic_categorize(self, prompt: str) -> str:
        """Fallback heuristic categorization with learned patterns"""
        prompt_lower = prompt.lower()
        
        # Enhanced categorization based on learned patterns
        if any(word in prompt_lower for word in ['ornate', 'elegant', 'gothic', 'paisley', 'decorative']):
            return "decorative_standard"
        elif any(word in prompt_lower for word in ['cylindrical', 'angular', 'geometric', 'diameter', 'threading']):
            return "technical_description"
        elif any(word in prompt_lower for word in ['silk', 'denim', 'cotton', 'fabric', 'clothing']):
            return "fashion_clothing"
        elif any(word in prompt_lower for word in ['flowing', 'ethereal', 'crystalline', 'artistic']):
            return "abstract_artistic"
        else:
            return "physical_object"
    
    def query_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        """Query DeepSeek-R1 via Ollama"""
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            raise Exception(f"DeepSeek query failed: {e}")
    
    def select_strategy_sequence(self, prompt: str, category: str, baseline_score: float) -> List[str]:
        """Select optimal sequence of strategies to try based on learned patterns and AI analysis"""
        
        # Apply learned insights from previous feedback
        strategies_to_try = []
        
        # Rule 1: If baseline score is high (>0.8), be very careful with modifications
        if baseline_score > 0.8:
            print(f"   🚨 High baseline score ({baseline_score:.3f}) - applying conservative strategy")
            if category == "decorative_standard":
                strategies_to_try = ["raw", "basic_description"]
            else:
                strategies_to_try = ["raw", "basic_description", "simplified_description"]
        
        # Rule 2: Category-specific preferences based on learned patterns
        elif category == "decorative_standard":
            strategies_to_try = ["raw", "basic_description", "artistic_focus"]
        elif category == "technical_description":
            strategies_to_try = ["material_focus", "minimal_enhancement", "concrete_object"]
        elif category == "fashion_clothing":
            strategies_to_try = ["basic_description", "minimal_enhancement", "simplified_description"]
        elif category == "abstract_artistic":
            strategies_to_try = ["geometric_focus", "artistic_focus", "current_production"]
        elif category == "physical_object":
            # Mixed results - use AI to decide
            strategies_to_try = ["material_focus", "raw", "concrete_object", "basic_description"]
        else:
            # Fallback
            strategies_to_try = ["material_focus", "raw", "basic_description"]
        
        # Rule 3: Use AI to refine strategy sequence if we have enough data
        if len(self.optimization_sessions) >= 3:
            try:
                # Get recent performance data
                recent_sessions = self.optimization_sessions[-5:]
                performance_summary = self.summarize_recent_performance(recent_sessions)
                
                system_prompt = f"""Based on recent optimization performance, recommend the best strategy sequence for this prompt.

Category: {category}
Baseline Score: {baseline_score:.3f}
Recent Performance Summary: {performance_summary}

Available Strategies: {list(self.base_strategies.keys())}
Current Sequence: {strategies_to_try}

Learned Patterns:
- Decorative objects work best with minimal changes (raw, basic_description)
- High-scoring prompts (>0.8) often get worse with heavy modifications
- Technical descriptions benefit from material_focus when problematic
- Physical objects have mixed results

Respond with a comma-separated list of 3-5 strategies in priority order."""

                user_prompt = f"Optimize strategy sequence for: '{prompt}'"
                
                ai_sequence = self.query_deepseek(system_prompt, user_prompt)
                
                # Parse AI response
                ai_strategies = [s.strip() for s in ai_sequence.split(',')]
                valid_ai_strategies = [s for s in ai_strategies if s in self.base_strategies]
                
                if len(valid_ai_strategies) >= 2:
                    print(f"   🤖 AI refined strategy sequence: {valid_ai_strategies}")
                    strategies_to_try = valid_ai_strategies
                    
            except Exception as e:
                print(f"   🤖 AI strategy refinement failed: {e}")
        
        # Limit to max attempts
        return strategies_to_try[:self.max_attempts]
    
    def summarize_recent_performance(self, sessions: List[OptimizationSession]) -> str:
        """Summarize recent performance for AI analysis"""
        if not sessions:
            return "No recent data"
        
        successful_sessions = [s for s in sessions if s.session_success]
        success_rate = len(successful_sessions) / len(sessions) * 100
        
        strategy_stats = {}
        for session in sessions:
            if session.best_attempt:
                strategy = session.best_attempt.strategy_name
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'count': 0, 'avg_improvement': 0}
                strategy_stats[strategy]['count'] += 1
                strategy_stats[strategy]['avg_improvement'] += session.session_improvement
        
        for strategy, stats in strategy_stats.items():
            stats['avg_improvement'] /= stats['count']
        
        return f"Success Rate: {success_rate:.1f}%, Strategy Performance: {strategy_stats}"
    
    def apply_strategy(self, prompt: str, strategy: str) -> str:
        """Apply optimization strategy to prompt"""
        if strategy in self.base_strategies:
            template = self.base_strategies[strategy]
            return template.format(prompt=prompt)
        else:
            return prompt
    
    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run production-accurate validation and return (validation_score, demo_fidelity)"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"❌ Validation failed for '{prompt[:50]}...'")
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0), data.get("demo_fidelity_score", 0.0)
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return 0.0, 0.0
    
    def optimize_prompt_multi_strategy(self, prompt: str) -> OptimizationSession:
        """Optimize a prompt using multiple strategies until improvement is found or limit reached"""
        
        print(f"\n🎯 MULTI-STRATEGY OPTIMIZATION: '{prompt}'")
        print("=" * 80)
        
        # Step 1: Categorize and get baseline
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        print(f"🧪 Getting baseline performance...")
        baseline_val, baseline_fidelity = self.run_validation(prompt)
        print(f"   📊 Baseline: Score={baseline_val:.4f}, Fidelity={baseline_fidelity:.3f}")
        
        # Step 2: Select strategy sequence
        strategy_sequence = self.select_strategy_sequence(prompt, category, baseline_val)
        print(f"🗂️ Strategy Sequence: {strategy_sequence}")
        
        # Step 3: Try strategies until improvement found or limit reached
        attempts = []
        best_attempt = None
        best_improvement = 0.0
        
        for attempt_num, strategy in enumerate(strategy_sequence, 1):
            print(f"\n[{attempt_num}/{len(strategy_sequence)}] Trying Strategy: {strategy}")
            
            # Apply strategy
            optimized_prompt = self.apply_strategy(prompt, strategy)
            print(f"   ✨ Optimized: '{optimized_prompt[:80]}{'...' if len(optimized_prompt) > 80 else ''}'")
            
            # Validate
            opt_val, opt_fidelity = self.run_validation(optimized_prompt)
            
            # Calculate improvements
            score_improvement = opt_val - baseline_val
            fidelity_improvement = opt_fidelity - baseline_fidelity
            
            print(f"   📊 Results: Score={opt_val:.4f} ({score_improvement:+.3f}), Fidelity={opt_fidelity:.3f} ({fidelity_improvement:+.3f})")
            
            # Store attempt
            attempt = OptimizationAttempt(
                strategy_name=strategy,
                optimized_prompt=optimized_prompt,
                validation_score=opt_val,
                demo_fidelity_score=opt_fidelity,
                score_improvement=score_improvement,
                fidelity_improvement=fidelity_improvement,
                attempt_number=attempt_num,
                timestamp=time.time()
            )
            attempts.append(attempt)
            
            # Check if this is the best attempt
            total_improvement = score_improvement + fidelity_improvement
            if total_improvement > best_improvement:
                best_improvement = total_improvement
                best_attempt = attempt
                print(f"   🌟 NEW BEST: Total improvement {total_improvement:+.3f}")
            
            # Early stopping if significant improvement found
            if total_improvement >= self.min_improvement_threshold:
                print(f"   ✅ SIGNIFICANT IMPROVEMENT FOUND! Stopping early.")
                break
            
            # Brief pause between attempts
            time.sleep(1)
        
        # Step 4: Create optimization session
        session_improvement = best_improvement if best_attempt else 0.0
        session_success = session_improvement >= self.min_improvement_threshold
        
        session = OptimizationSession(
            original_prompt=prompt,
            prompt_category=category,
            baseline_score=baseline_val,
            baseline_fidelity=baseline_fidelity,
            attempts=attempts,
            best_attempt=best_attempt,
            total_attempts=len(attempts),
            session_improvement=session_improvement,
            session_success=session_success,
            timestamp=time.time()
        )
        
        print(f"\n📊 SESSION SUMMARY:")
        print(f"   Total Attempts: {session.total_attempts}")
        print(f"   Best Strategy: {session.best_attempt.strategy_name if session.best_attempt else 'None'}")
        print(f"   Session Improvement: {session.session_improvement:+.3f}")
        print(f"   Success: {'✅' if session.session_success else '❌'}")
        
        # Store session
        self.optimization_sessions.append(session)
        self.store_session_in_db(session)
        
        return session
    
    def store_session_in_db(self, session: OptimizationSession):
        """Store optimization session and attempts in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store session
        cursor.execute('''
            INSERT INTO optimization_sessions 
            (original_prompt, prompt_category, baseline_score, baseline_fidelity,
             total_attempts, best_strategy, best_score, best_fidelity,
             session_improvement, session_success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.original_prompt, session.prompt_category, session.baseline_score,
            session.baseline_fidelity, session.total_attempts,
            session.best_attempt.strategy_name if session.best_attempt else None,
            session.best_attempt.validation_score if session.best_attempt else 0.0,
            session.best_attempt.demo_fidelity_score if session.best_attempt else 0.0,
            session.session_improvement, session.session_success, session.timestamp
        ))
        
        session_id = cursor.lastrowid
        
        # Store attempts
        for attempt in session.attempts:
            cursor.execute('''
                INSERT INTO optimization_attempts
                (session_id, strategy_name, optimized_prompt, validation_score,
                 demo_fidelity_score, score_improvement, fidelity_improvement,
                 attempt_number, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, attempt.strategy_name, attempt.optimized_prompt,
                attempt.validation_score, attempt.demo_fidelity_score,
                attempt.score_improvement, attempt.fidelity_improvement,
                attempt.attempt_number, attempt.timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def run_enhanced_learning_session(self, test_prompts: List[str]):
        """Run enhanced optimization session with multi-strategy testing"""
        
        print("🧠 ENHANCED ADAPTIVE LEARNING SESSION v2.0")
        print("=" * 80)
        print(f"📚 Testing {len(test_prompts)} prompts with multi-strategy optimization")
        print(f"🎯 Max attempts per prompt: {self.max_attempts}")
        print(f"📊 Improvement threshold: {self.min_improvement_threshold}")
        print(f"🤖 Model: {self.model_name}")
        print("=" * 80)
        
        all_sessions = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Processing prompt {i}")
            
            session = self.optimize_prompt_multi_strategy(prompt)
            all_sessions.append(session)
            
            # Update strategy performance tracking
            self.update_strategy_performance(session)
            
            # Brief pause between prompts
            time.sleep(2)
        
        # Final analysis
        self.run_final_analysis(all_sessions)
        
        return all_sessions
    
    def update_strategy_performance(self, session: OptimizationSession):
        """Update strategy performance tracking based on session results"""
        category = session.prompt_category
        
        for attempt in session.attempts:
            strategy = attempt.strategy_name
            
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {}
            
            if category not in self.strategy_performance[strategy]:
                self.strategy_performance[strategy][category] = {
                    'improvements': [],
                    'usage_count': 0
                }
            
            self.strategy_performance[strategy][category]['improvements'].append(
                attempt.score_improvement + attempt.fidelity_improvement
            )
            self.strategy_performance[strategy][category]['usage_count'] += 1
    
    def run_final_analysis(self, sessions: List[OptimizationSession]):
        """Run comprehensive final analysis with AI insights"""
        
        print(f"\n🎓 FINAL ENHANCED SESSION ANALYSIS")
        print("=" * 80)
        
        # Calculate statistics
        total_sessions = len(sessions)
        successful_sessions = [s for s in sessions if s.session_success]
        success_rate = (len(successful_sessions) / total_sessions) * 100 if total_sessions > 0 else 0
        
        total_attempts = sum(s.total_attempts for s in sessions)
        avg_attempts = total_attempts / total_sessions if total_sessions > 0 else 0
        
        improvements = [s.session_improvement for s in sessions]
        avg_improvement = statistics.mean(improvements) if improvements else 0
        
        print(f"📊 ENHANCED SESSION STATISTICS:")
        print(f"   Total Prompts: {total_sessions}")
        print(f"   Successful Optimizations: {len(successful_sessions)}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Strategy Attempts: {total_attempts}")
        print(f"   Average Attempts per Prompt: {avg_attempts:.1f}")
        print(f"   Average Improvement: {avg_improvement:+.3f}")
        
        # Strategy effectiveness analysis
        print(f"\n🔧 STRATEGY EFFECTIVENESS:")
        strategy_success_counts = {}
        for session in sessions:
            if session.best_attempt:
                strategy = session.best_attempt.strategy_name
                if strategy not in strategy_success_counts:
                    strategy_success_counts[strategy] = 0
                strategy_success_counts[strategy] += 1
        
        for strategy, count in sorted(strategy_success_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_sessions) * 100
            print(f"   {strategy}: {count}/{total_sessions} ({percentage:.1f}%)")
        
        # AI-powered insights
        try:
            ai_insights = self.generate_ai_insights(sessions)
            print(f"\n🤖 AI-POWERED INSIGHTS:")
            print(ai_insights)
        except Exception as e:
            print(f"🤖 AI insights generation failed: {e}")
        
        # Save comprehensive results
        session_data = {
            "enhanced_session_stats": {
                "total_sessions": total_sessions,
                "successful_sessions": len(successful_sessions),
                "success_rate": success_rate,
                "total_attempts": total_attempts,
                "avg_attempts_per_prompt": avg_attempts,
                "average_improvement": avg_improvement,
                "strategy_success_counts": strategy_success_counts,
                "timestamp": time.time()
            },
            "all_sessions": [asdict(s) for s in sessions],
            "strategy_performance": self.strategy_performance
        }
        
        output_file = f"enhanced_adaptive_session_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\n💾 Enhanced session data saved to: {output_file}")
    
    def generate_ai_insights(self, sessions: List[OptimizationSession]) -> str:
        """Generate AI-powered insights from session results"""
        
        # Prepare summary for AI
        session_summary = []
        for session in sessions:
            session_summary.append({
                "prompt": session.original_prompt[:50],
                "category": session.prompt_category,
                "baseline_score": session.baseline_score,
                "attempts": session.total_attempts,
                "best_strategy": session.best_attempt.strategy_name if session.best_attempt else None,
                "improvement": session.session_improvement,
                "success": session.session_success
            })
        
        system_prompt = """Analyze these multi-strategy optimization results and provide key insights.

Focus on:
1. Which strategies consistently work best
2. Pattern recognition for when to use multi-strategy vs single-strategy
3. Category-specific insights
4. Recommendations for improving the optimization process

Be specific and actionable."""

        user_prompt = f"Analyze these enhanced optimization results:\n{json.dumps(session_summary, indent=2)}"
        
        return self.query_deepseek(system_prompt, user_prompt)

def main():
    """Run enhanced adaptive learning with multi-strategy optimization"""
    
    # Diverse test prompts
    test_prompts = [
        # Previously problematic prompts
        "hexagonal prism steel structure",
        "cylindrical copper pipe diameter 5cm",
        "wooden kitchen spoon carved handle",
        "elegant silk scarf paisley pattern",
        "ornate silver candelabra gothic design",
        
        # New challenging prompts
        "industrial gear mechanism bronze",
        "delicate porcelain tea cup floral",
        "modern minimalist lamp geometric",
        "vintage leather briefcase worn",
        "crystal wine glass etched pattern"
    ]
    
    # Initialize enhanced optimizer
    optimizer = AdaptiveLearningOptimizerV2(max_attempts_per_prompt=4, min_improvement_threshold=0.05)
    
    # Run enhanced learning session
    sessions = optimizer.run_enhanced_learning_session(test_prompts)
    
    print(f"\n🎯 ENHANCED ADAPTIVE LEARNING COMPLETE!")
    print(f"Tested {len(sessions)} prompts with multi-strategy optimization")
    print(f"Average attempts per prompt: {sum(s.total_attempts for s in sessions) / len(sessions):.1f}")
    print(f"Success rate: {sum(1 for s in sessions if s.session_success) / len(sessions) * 100:.1f}%")

if __name__ == "__main__":
    main() 
