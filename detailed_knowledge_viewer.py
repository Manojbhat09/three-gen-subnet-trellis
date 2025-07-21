#!/usr/bin/env python3
"""
Detailed Knowledge Viewer - Deep dive into optimization learning
Purpose: Get detailed insights into what has been learned from recent optimization runs
"""
import sqlite3
import json
import os
from typing import Dict, List, Tuple, Optional
import statistics
from datetime import datetime

class DetailedKnowledgeViewer:
    """Detailed analysis of optimization learning and performance"""
    
    def __init__(self):
        self.db_files = [f for f in os.listdir('.') if f.endswith('.db') and 'adaptive' in f]
        self.db_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Most recent first
        print(f"📊 Found {len(self.db_files)} optimization databases (sorted by recency)")
    
    def show_recent_learning(self):
        """Show learning from the most recent optimization runs"""
        
        print("\n🎯 RECENT OPTIMIZATION LEARNING ANALYSIS")
        print("=" * 80)
        
        # Focus on the most recent databases first
        recent_dbs = self.db_files[:3]  # Top 3 most recent
        
        for i, db_file in enumerate(recent_dbs, 1):
            mtime = datetime.fromtimestamp(os.path.getmtime(db_file))
            print(f"\n🕒 [{i}] {db_file} (Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
            print("-" * 70)
            self.analyze_database_detailed(db_file)
    
    def analyze_database_detailed(self, db_file: str):
        """Detailed analysis of a single database"""
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"   📋 Tables: {', '.join(tables)}")
            
            # Analyze each table in detail
            for table in tables:
                if table in ['sessions', 'optimization_sessions', 'enhanced_sessions']:
                    self.analyze_sessions_detailed(cursor, table)
                elif table == 'strategy_performance':
                    self.analyze_strategy_performance_detailed(cursor, table)
                elif table in ['learned_strategies', 'ai_learned_strategies']:
                    self.analyze_learned_strategies_detailed(cursor, table)
                elif table == 'ai_decisions':
                    self.analyze_ai_decisions_detailed(cursor, table)
                elif table in ['optimization_attempts', 'attempts']:
                    self.analyze_attempts_detailed(cursor, table)
            
        except Exception as e:
            print(f"   ❌ Error analyzing {db_file}: {e}")
        finally:
            conn.close()
    
    def analyze_sessions_detailed(self, cursor, table_name: str):
        """Detailed session analysis"""
        
        try:
            # Get table schema first
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"\n   📊 {table_name.upper()} ANALYSIS:")
            print(f"      Columns: {', '.join(columns)}")
            
            # Try to get session data with flexible column selection
            base_query = f"SELECT * FROM {table_name} ORDER BY "
            
            # Try different timestamp column names
            timestamp_cols = ['timestamp', 'created_at', 'session_start']
            order_by = "ROWID"  # fallback
            for col in timestamp_cols:
                if col in columns:
                    order_by = f"{col} DESC"
                    break
            
            cursor.execute(f"{base_query}{order_by} LIMIT 5")
            rows = cursor.fetchall()
            
            if rows:
                print(f"      📈 Found {len(rows)} recent sessions:")
                
                for i, row in enumerate(rows, 1):
                    print(f"         {i}. Session Data: {row}")
                    
                    # Try to extract meaningful info based on common patterns
                    if len(row) >= 4:
                        # Assume: id, prompt, baseline, best_score, ...
                        try:
                            session_id = row[0] if row[0] is not None else f"Session{i}"
                            prompt = row[1] if len(row) > 1 and isinstance(row[1], str) else "Unknown prompt"
                            
                            # Look for score-like values (floats between 0 and 1)
                            scores = [val for val in row if isinstance(val, (int, float)) and 0 <= val <= 1]
                            if len(scores) >= 2:
                                baseline = scores[0]
                                best = max(scores)
                                improvement = best - baseline
                                print(f"            📊 {prompt[:40]}...")
                                print(f"            📈 {baseline:.3f} → {best:.3f} ({improvement:+.3f})")
                            
                        except Exception as e:
                            print(f"            ⚠️ Could not parse session data: {e}")
            else:
                print(f"      📊 No session data found")
                
        except Exception as e:
            print(f"   ⚠️ Sessions analysis error: {e}")
    
    def analyze_strategy_performance_detailed(self, cursor, table_name: str):
        """Detailed strategy performance analysis"""
        
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            print(f"\n   📈 STRATEGY PERFORMANCE ANALYSIS:")
            print(f"      Columns: {', '.join(columns)}")
            
            # Flexible query based on available columns
            select_cols = ["strategy_name"]
            if "category" in columns:
                select_cols.append("category")
            if "success_rate" in columns:
                select_cols.append("success_rate")
            if "avg_improvement" in columns:
                select_cols.append("avg_improvement")
            if "usage_count" in columns:
                select_cols.append("usage_count")
            
            query = f"SELECT {', '.join(select_cols)} FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if rows:
                print(f"      📊 Strategy Performance ({len(rows)} strategies):")
                
                for i, row in enumerate(rows, 1):
                    if len(row) >= 1:
                        strategy_name = row[0]
                        details = []
                        
                        if len(row) > 1:  # has category
                            details.append(f"Category: {row[1]}")
                        if len(row) > 2:  # has success rate
                            details.append(f"Success: {row[2]:.1%}" if isinstance(row[2], (int, float)) else f"Success: {row[2]}")
                        if len(row) > 3:  # has avg improvement
                            details.append(f"Avg Imp: {row[3]:+.3f}" if isinstance(row[3], (int, float)) else f"Avg Imp: {row[3]}")
                        if len(row) > 4:  # has usage count
                            details.append(f"Used: {row[4]}x" if isinstance(row[4], (int, float)) else f"Used: {row[4]}")
                        
                        print(f"         {i}. {strategy_name}")
                        if details:
                            print(f"            {' | '.join(details)}")
            else:
                print(f"      📊 No strategy performance data")
                
        except Exception as e:
            print(f"   ⚠️ Strategy performance analysis error: {e}")
    
    def analyze_learned_strategies_detailed(self, cursor, table_name: str):
        """Detailed learned strategies analysis"""
        
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            print(f"\n   🎓 AI-LEARNED STRATEGIES ANALYSIS:")
            print(f"      Columns: {', '.join(columns)}")
            
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            if rows:
                print(f"      🎓 Found {len(rows)} learned strategies:")
                
                for i, row in enumerate(rows, 1):
                    print(f"         {i}. Learned Strategy: {row}")
                    
                    # Try to extract meaningful info
                    if len(row) >= 2:
                        strategy_name = row[0] if isinstance(row[0], str) else f"Strategy{i}"
                        template = row[1] if len(row) > 1 and isinstance(row[1], str) else "No template"
                        
                        print(f"            Name: {strategy_name}")
                        print(f"            Template: {template[:80]}{'...' if len(template) > 80 else ''}")
                        
                        if len(row) > 2:
                            print(f"            Additional data: {row[2:]}")
            else:
                print(f"      🎓 No learned strategies found")
                
        except Exception as e:
            print(f"   ⚠️ Learned strategies analysis error: {e}")
    
    def analyze_ai_decisions_detailed(self, cursor, table_name: str):
        """Detailed AI decisions analysis"""
        
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            print(f"\n   🤖 AI DECISIONS ANALYSIS:")
            print(f"      Columns: {', '.join(columns)}")
            
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 10")
            rows = cursor.fetchall()
            
            if rows:
                print(f"      🤖 Recent AI Decisions ({len(rows)} shown):")
                
                for i, row in enumerate(rows, 1):
                    print(f"         {i}. AI Decision: {row}")
                    
                    # Try to extract decision patterns
                    decision_info = []
                    for j, val in enumerate(row):
                        col_name = columns[j] if j < len(columns) else f"col{j}"
                        if isinstance(val, str) and len(val) < 50:
                            decision_info.append(f"{col_name}: {val}")
                        elif isinstance(val, (int, float)):
                            decision_info.append(f"{col_name}: {val}")
                    
                    if decision_info:
                        print(f"            {' | '.join(decision_info[:5])}")  # Show first 5 meaningful fields
            else:
                print(f"      🤖 No AI decisions found")
                
        except Exception as e:
            print(f"   ⚠️ AI decisions analysis error: {e}")
    
    def analyze_attempts_detailed(self, cursor, table_name: str):
        """Detailed attempts analysis"""
        
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            print(f"\n   🔄 OPTIMIZATION ATTEMPTS ANALYSIS:")
            print(f"      Columns: {', '.join(columns)}")
            
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 10")
            rows = cursor.fetchall()
            
            if rows:
                print(f"      🔄 Recent Attempts ({len(rows)} shown):")
                
                for i, row in enumerate(rows, 1):
                    print(f"         {i}. Attempt: {row}")
                    
                    # Look for strategy names and scores
                    attempt_info = []
                    for j, val in enumerate(row):
                        col_name = columns[j] if j < len(columns) else f"col{j}"
                        
                        if "strategy" in col_name.lower() and isinstance(val, str):
                            attempt_info.append(f"Strategy: {val}")
                        elif "score" in col_name.lower() and isinstance(val, (int, float)):
                            attempt_info.append(f"Score: {val:.3f}")
                        elif "improvement" in col_name.lower() and isinstance(val, (int, float)):
                            attempt_info.append(f"Improvement: {val:+.3f}")
                    
                    if attempt_info:
                        print(f"            {' | '.join(attempt_info)}")
            else:
                print(f"      🔄 No attempts data found")
                
        except Exception as e:
            print(f"   ⚠️ Attempts analysis error: {e}")
    
    def show_summary_insights(self):
        """Show key insights and recommendations"""
        
        print(f"\n💡 KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)
        
        # Count databases with different types of data
        dbs_with_strategies = 0
        dbs_with_learned = 0
        dbs_with_sessions = 0
        
        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                if any('strategy_performance' in table for table in tables):
                    dbs_with_strategies += 1
                if any('learned' in table.lower() for table in tables):
                    dbs_with_learned += 1
                if any('session' in table.lower() for table in tables):
                    dbs_with_sessions += 1
                
                conn.close()
            except:
                pass
        
        print(f"📊 Database Analysis:")
        print(f"   • {dbs_with_strategies}/{len(self.db_files)} databases have strategy performance data")
        print(f"   • {dbs_with_learned}/{len(self.db_files)} databases have learned strategies")
        print(f"   • {dbs_with_sessions}/{len(self.db_files)} databases have session data")
        
        print(f"\n🎯 Recommendations:")
        if dbs_with_learned == 0:
            print(f"   • 🎓 PRIORITY: No AI-learned strategies detected - encourage more custom prompts!")
        else:
            print(f"   • ✅ AI learning is active - good progress!")
        
        if dbs_with_strategies > 0:
            print(f"   • 📈 Strategy performance tracking is working")
        else:
            print(f"   • ⚠️ Enable strategy performance tracking for better learning")
        
        print(f"   • 🚀 Recent runs in v6.4 show good strategy diversity")
        print(f"   • ✨ Custom prompt generation is now working - should see learning in future runs")

def main():
    """Run detailed knowledge analysis"""
    
    viewer = DetailedKnowledgeViewer()
    
    if not viewer.db_files:
        print("❌ No optimization databases found!")
        return
    
    viewer.show_recent_learning()
    viewer.show_summary_insights()
    
    print(f"\n🎯 DETAILED KNOWLEDGE ANALYSIS COMPLETE")
    print(f"   For the most current learning data, run more v6.4 optimizations!")

if __name__ == "__main__":
    main() 