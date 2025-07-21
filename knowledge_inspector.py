#!/usr/bin/env python3
"""
Knowledge Inspector - Analyze Learning from Optimization Runs
Purpose: Inspect and analyze what the AI has learned from optimization sessions
"""
import sqlite3
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics
from datetime import datetime

@dataclass
class StrategyPerformance:
    name: str
    category: str
    success_rate: float
    avg_improvement: float
    usage_count: int
    last_used: Optional[float] = None

@dataclass
class LearnedStrategy:
    name: str
    template: str
    category_affinity: str
    base_success_rate: float
    usage_count: int
    learned_from_prompt: str
    timestamp: Optional[float] = None

@dataclass
class SessionSummary:
    id: int
    original_prompt: str
    category: str
    baseline_score: float
    best_score: float
    improvement: float
    total_attempts: int
    reached_target: bool
    reached_ultra: bool
    timestamp: float

class KnowledgeInspector:
    """Tool to inspect learning from optimization databases"""
    
    def __init__(self):
        self.db_files = [
            "adaptive_optimizer_v6_1.db",
            "adaptive_optimizer_v6_3.db", 
            "adaptive_optimizer_v6_4.db",
            "adaptive_learning_v2.db",
            "advanced_adaptive_optimizer.db",
            "ai_recommendation_engine.db",
            "enhanced_ai_optimizer_v4.db"
        ]
        
        # Find existing databases
        self.existing_dbs = [db for db in self.db_files if os.path.exists(db)]
        print(f"📊 Found {len(self.existing_dbs)} optimization databases")
    
    def inspect_all_knowledge(self):
        """Comprehensive knowledge inspection across all databases"""
        
        print("\n🧠 COMPREHENSIVE KNOWLEDGE INSPECTION")
        print("=" * 80)
        
        for db_file in self.existing_dbs:
            print(f"\n📁 Analyzing: {db_file}")
            print("-" * 60)
            self.inspect_database(db_file)
        
        print(f"\n📋 CONSOLIDATED KNOWLEDGE SUMMARY")
        print("=" * 80)
        self.create_consolidated_summary()
    
    def inspect_database(self, db_file: str):
        """Inspect a single database"""
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"   📊 Tables found: {', '.join(tables)}")
            
            # Strategy performance analysis
            if "strategy_performance" in tables:
                self.analyze_strategy_performance(cursor, db_file)
            
            # Learned strategies analysis
            if "learned_strategies" in tables:
                self.analyze_learned_strategies(cursor, db_file)
            
            # Sessions analysis
            if "sessions" in tables:
                self.analyze_sessions(cursor, db_file)
            
            # AI decisions analysis
            if "ai_decisions" in tables:
                self.analyze_ai_decisions(cursor, db_file)
            
        except Exception as e:
            print(f"   ❌ Error analyzing {db_file}: {e}")
        finally:
            conn.close()
    
    def analyze_strategy_performance(self, cursor, db_name: str):
        """Analyze strategy performance data"""
        
        try:
            cursor.execute("""
                SELECT strategy_name, category, success_rate, avg_improvement, usage_count, last_used 
                FROM strategy_performance 
                ORDER BY success_rate DESC, avg_improvement DESC
            """)
            
            strategies = []
            for row in cursor.fetchall():
                name, cat, rate, imp, count, last_used = row
                strategies.append(StrategyPerformance(name, cat, rate, imp, count, last_used))
            
            if strategies:
                print(f"\n   📈 STRATEGY PERFORMANCE ({len(strategies)} strategies):")
                
                # Top performers
                top_strategies = sorted(strategies, key=lambda s: (s.success_rate, s.avg_improvement), reverse=True)[:5]
                for i, strategy in enumerate(top_strategies, 1):
                    print(f"      {i}. {strategy.name} ({strategy.category})")
                    print(f"         Success: {strategy.success_rate:.1%} | Avg Improvement: {strategy.avg_improvement:+.3f} | Used: {strategy.usage_count}x")
                
                # Category breakdown
                categories = {}
                for strategy in strategies:
                    if strategy.category not in categories:
                        categories[strategy.category] = []
                    categories[strategy.category].append(strategy)
                
                print(f"\n   📊 BY CATEGORY:")
                for category, cat_strategies in categories.items():
                    avg_success = statistics.mean([s.success_rate for s in cat_strategies])
                    avg_improvement = statistics.mean([s.avg_improvement for s in cat_strategies])
                    total_usage = sum([s.usage_count for s in cat_strategies])
                    print(f"      {category}: {len(cat_strategies)} strategies, {avg_success:.1%} avg success, {avg_improvement:+.3f} avg improvement, {total_usage} total uses")
                
            else:
                print(f"   📊 No strategy performance data found")
                
        except Exception as e:
            print(f"   ⚠️ Strategy performance analysis error: {e}")
    
    def analyze_learned_strategies(self, cursor, db_name: str):
        """Analyze AI-learned strategies"""
        
        try:
            cursor.execute("""
                SELECT strategy_name, template, category_affinity, base_success_rate, usage_count, learned_from_prompt, timestamp
                FROM learned_strategies 
                ORDER BY base_success_rate DESC, usage_count DESC
            """)
            
            learned = []
            for row in cursor.fetchall():
                name, template, cat, rate, count, prompt, timestamp = row
                learned.append(LearnedStrategy(name, template, cat, rate, count, prompt, timestamp))
            
            if learned:
                print(f"\n   🎓 AI-LEARNED STRATEGIES ({len(learned)} strategies):")
                
                for i, strategy in enumerate(learned, 1):
                    timestamp_str = datetime.fromtimestamp(strategy.timestamp).strftime("%Y-%m-%d %H:%M") if strategy.timestamp else "Unknown"
                    print(f"      {i}. {strategy.name}")
                    print(f"         Template: {strategy.template}")
                    print(f"         Category: {strategy.category_affinity} | Success: {strategy.base_success_rate:.1%} | Used: {strategy.usage_count}x")
                    print(f"         Learned from: '{strategy.learned_from_prompt}' | When: {timestamp_str}")
                    print()
                
            else:
                print(f"   🎓 No AI-learned strategies found")
                
        except Exception as e:
            print(f"   ⚠️ Learned strategies analysis error: {e}")
    
    def analyze_sessions(self, cursor, db_name: str):
        """Analyze optimization sessions"""
        
        try:
            # Try different possible session table schemas
            schemas_to_try = [
                "SELECT id, original_prompt, category, baseline_score, best_score, session_improvement, total_attempts, reached_target, reached_ultra, timestamp FROM sessions",
                "SELECT id, original_prompt, category, baseline_score, best_score, total_attempts, reached_target, reached_ultra, timestamp FROM sessions",
                "SELECT * FROM sessions"
            ]
            
            sessions = []
            for schema in schemas_to_try:
                try:
                    cursor.execute(schema)
                    rows = cursor.fetchall()
                    
                    if rows:
                        for row in rows:
                            if len(row) >= 9:
                                id, prompt, cat, baseline, best, improvement, attempts, target, ultra, timestamp = row[:10]
                                sessions.append(SessionSummary(id, prompt, cat, baseline, best, improvement, attempts, bool(target), bool(ultra), timestamp))
                            else:
                                # Fallback for different schema
                                print(f"   📊 Session data found but schema differs: {len(row)} columns")
                        break
                except:
                    continue
            
            if sessions:
                print(f"\n   📈 OPTIMIZATION SESSIONS ({len(sessions)} sessions):")
                
                # Overall stats
                total_improvements = [s.improvement for s in sessions if s.improvement > 0]
                avg_improvement = statistics.mean(total_improvements) if total_improvements else 0
                success_rate = len(total_improvements) / len(sessions) if sessions else 0
                reached_target = sum(1 for s in sessions if s.reached_target)
                reached_ultra = sum(1 for s in sessions if s.reached_ultra)
                
                print(f"      📊 Overall Performance:")
                print(f"         Success Rate: {success_rate:.1%} ({len(total_improvements)}/{len(sessions)} sessions improved)")
                print(f"         Average Improvement: {avg_improvement:+.3f}")
                print(f"         Reached Target: {reached_target}/{len(sessions)} ({reached_target/len(sessions)*100:.1f}%)")
                print(f"         Reached Ultra: {reached_ultra}/{len(sessions)} ({reached_ultra/len(sessions)*100:.1f}%)")
                
                # Recent sessions
                recent_sessions = sorted(sessions, key=lambda s: s.timestamp, reverse=True)[:3]
                print(f"\n      🕒 Recent Sessions:")
                for i, session in enumerate(recent_sessions, 1):
                    timestamp_str = datetime.fromtimestamp(session.timestamp).strftime("%Y-%m-%d %H:%M")
                    print(f"         {i}. '{session.original_prompt}' ({session.category})")
                    print(f"            {session.baseline_score:.3f} → {session.best_score:.3f} ({session.improvement:+.3f}) | {session.total_attempts} attempts | {timestamp_str}")
                    print(f"            Target: {'✅' if session.reached_target else '❌'} | Ultra: {'✅' if session.reached_ultra else '❌'}")
                
            else:
                print(f"   📈 No session data found")
                
        except Exception as e:
            print(f"   ⚠️ Sessions analysis error: {e}")
    
    def analyze_ai_decisions(self, cursor, db_name: str):
        """Analyze AI decision patterns"""
        
        try:
            cursor.execute("""
                SELECT persona_used, decision_type, confidence, led_to_improvement, contributed_to_best
                FROM ai_decisions
            """)
            
            decisions = cursor.fetchall()
            
            if decisions:
                print(f"\n   🤖 AI DECISION ANALYSIS ({len(decisions)} decisions):")
                
                # Persona performance
                personas = {}
                for persona, dtype, conf, improved, best in decisions:
                    if persona not in personas:
                        personas[persona] = {"total": 0, "improved": 0, "best": 0, "avg_confidence": []}
                    personas[persona]["total"] += 1
                    if improved:
                        personas[persona]["improved"] += 1
                    if best:
                        personas[persona]["best"] += 1
                    personas[persona]["avg_confidence"].append(conf)
                
                print(f"      🎭 Persona Performance:")
                for persona, stats in personas.items():
                    success_rate = stats["improved"] / stats["total"] if stats["total"] > 0 else 0
                    best_rate = stats["best"] / stats["total"] if stats["total"] > 0 else 0
                    avg_conf = statistics.mean(stats["avg_confidence"]) if stats["avg_confidence"] else 0
                    print(f"         {persona}: {success_rate:.1%} success ({stats['improved']}/{stats['total']}) | {best_rate:.1%} best | {avg_conf:.2f} avg confidence")
                
                # Decision type performance
                decision_types = {}
                for persona, dtype, conf, improved, best in decisions:
                    if dtype not in decision_types:
                        decision_types[dtype] = {"total": 0, "improved": 0, "best": 0}
                    decision_types[dtype]["total"] += 1
                    if improved:
                        decision_types[dtype]["improved"] += 1
                    if best:
                        decision_types[dtype]["best"] += 1
                
                print(f"\n      🎯 Decision Type Performance:")
                for dtype, stats in decision_types.items():
                    success_rate = stats["improved"] / stats["total"] if stats["total"] > 0 else 0
                    best_rate = stats["best"] / stats["total"] if stats["total"] > 0 else 0
                    print(f"         {dtype}: {success_rate:.1%} success ({stats['improved']}/{stats['total']}) | {best_rate:.1%} contributed to best")
                
            else:
                print(f"   🤖 No AI decision data found")
                
        except Exception as e:
            print(f"   ⚠️ AI decisions analysis error: {e}")
    
    def create_consolidated_summary(self):
        """Create a consolidated summary across all databases"""
        
        all_strategies = {}
        all_learned = []
        all_sessions = []
        
        for db_file in self.existing_dbs:
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Collect strategy performance
                try:
                    cursor.execute("SELECT strategy_name, category, success_rate, avg_improvement, usage_count FROM strategy_performance")
                    for row in cursor.fetchall():
                        name, cat, rate, imp, count = row
                        key = f"{name}_{cat}"
                        if key not in all_strategies:
                            all_strategies[key] = {"name": name, "category": cat, "total_usage": 0, "success_rates": [], "improvements": []}
                        all_strategies[key]["total_usage"] += count
                        all_strategies[key]["success_rates"].append(rate)
                        all_strategies[key]["improvements"].append(imp)
                except:
                    pass
                
                # Collect learned strategies
                try:
                    cursor.execute("SELECT strategy_name, template, category_affinity, base_success_rate FROM learned_strategies")
                    for row in cursor.fetchall():
                        all_learned.append(row)
                except:
                    pass
                
                # Collect sessions
                try:
                    cursor.execute("SELECT original_prompt, category, baseline_score, best_score FROM sessions")
                    for row in cursor.fetchall():
                        all_sessions.append(row)
                except:
                    pass
                
                conn.close()
                
            except Exception as e:
                print(f"   ⚠️ Error processing {db_file}: {e}")
        
        # Print consolidated analysis
        if all_strategies:
            print(f"\n📊 CONSOLIDATED STRATEGY PERFORMANCE:")
            
            # Calculate overall performance
            strategy_summary = []
            for key, data in all_strategies.items():
                avg_success = statistics.mean(data["success_rates"]) if data["success_rates"] else 0
                avg_improvement = statistics.mean(data["improvements"]) if data["improvements"] else 0
                strategy_summary.append((data["name"], data["category"], avg_success, avg_improvement, data["total_usage"]))
            
            # Sort by performance
            strategy_summary.sort(key=lambda x: (x[2], x[3]), reverse=True)
            
            print(f"   🏆 TOP PERFORMING STRATEGIES (across all databases):")
            for i, (name, cat, rate, imp, usage) in enumerate(strategy_summary[:10], 1):
                print(f"      {i}. {name} ({cat}): {rate:.1%} success, {imp:+.3f} avg improvement, {usage} total uses")
        
        if all_learned:
            print(f"\n🎓 TOTAL AI-LEARNED STRATEGIES: {len(all_learned)}")
            for name, template, cat, rate in all_learned:
                print(f"   - {name} ({cat}): {rate:.1%} success")
                print(f"     Template: {template}")
        
        if all_sessions:
            print(f"\n📈 CONSOLIDATED SESSION ANALYSIS:")
            improvements = []
            categories = {}
            
            for prompt, cat, baseline, best in all_sessions:
                improvement = best - baseline
                improvements.append(improvement)
                
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(improvement)
            
            if improvements:
                avg_improvement = statistics.mean(improvements)
                success_rate = len([i for i in improvements if i > 0]) / len(improvements)
                print(f"   📊 Overall: {success_rate:.1%} success rate, {avg_improvement:+.3f} average improvement")
                
                print(f"   📋 By Category:")
                for cat, cat_improvements in categories.items():
                    cat_avg = statistics.mean(cat_improvements)
                    cat_success = len([i for i in cat_improvements if i > 0]) / len(cat_improvements)
                    print(f"      {cat}: {cat_success:.1%} success, {cat_avg:+.3f} avg improvement ({len(cat_improvements)} sessions)")
        
        # Recommendations
        print(f"\n💡 LEARNING INSIGHTS & RECOMMENDATIONS:")
        
        if strategy_summary:
            best_strategy = strategy_summary[0]
            print(f"   🏆 Best Overall Strategy: {best_strategy[0]} ({best_strategy[1]}) - {best_strategy[2]:.1%} success rate")
        
        if all_learned:
            print(f"   🎓 AI Learning Active: {len(all_learned)} custom strategies learned")
        else:
            print(f"   🎓 AI Learning Opportunity: No custom strategies learned yet - encourage more WRITE_CUSTOM decisions")
        
        custom_prompt_usage = sum(1 for name, cat, rate, imp, usage in strategy_summary if "custom" in name.lower())
        if custom_prompt_usage > 0:
            print(f"   ✨ Custom Prompt Usage: {custom_prompt_usage} custom strategies being used")
        else:
            print(f"   ✨ Custom Prompt Opportunity: Increase custom prompt generation for breakthroughs")

def main():
    """Run comprehensive knowledge inspection"""
    
    inspector = KnowledgeInspector()
    
    if not inspector.existing_dbs:
        print("❌ No optimization databases found!")
        print("   Run some optimization sessions first to generate learning data.")
        return
    
    inspector.inspect_all_knowledge()
    
    print(f"\n🎯 KNOWLEDGE INSPECTION COMPLETE")
    print(f"   Analyzed {len(inspector.existing_dbs)} databases")
    print(f"   Use this knowledge to improve future optimization runs!")

if __name__ == "__main__":
    main() 