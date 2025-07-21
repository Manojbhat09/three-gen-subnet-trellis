#!/usr/bin/env python3
"""
DeepSeek AI Analyst & Policy Manager
====================================
Purpose: AI-powered analysis of optimization patterns and automatic policy creation
Features:
- DeepSeek analyzes optimization results and extracts patterns
- Automatic policy generation for reusable prompt optimization
- AI-powered insights and recommendations
- Learning pattern extraction and storage
"""

import requests
import json
import time
import sqlite3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics
import re

@dataclass
class OptimizationPattern:
    pattern_name: str
    category: str
    success_rate: float
    avg_score: float
    max_score: float
    pattern_template: str
    success_conditions: List[str]
    failure_conditions: List[str]
    ai_insights: str
    usage_count: int
    last_updated: float

@dataclass
class PolicyRecommendation:
    policy_name: str
    target_category: str
    recommended_template: str
    confidence_score: float
    expected_performance: float
    application_conditions: List[str]
    ai_reasoning: str
    timestamp: float

class DeepSeekAIAnalyst:
    """AI-powered pattern analysis and policy generation system"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "ai_pattern_policies.db"
        
        self.setup_policy_database()
        
        print("🧠 DEEPSEEK AI ANALYST & POLICY MANAGER")
        print("🎯 Mission: AI-powered pattern analysis and policy creation")
        print("⚡ Features: Automatic insights, pattern extraction, policy generation")
        print("=" * 80)

    def setup_policy_database(self):
        """Setup database for pattern policies and AI insights"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Optimization patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE,
                category TEXT,
                success_rate REAL,
                avg_score REAL,
                max_score REAL,
                pattern_template TEXT,
                success_conditions TEXT,
                failure_conditions TEXT,
                ai_insights TEXT,
                usage_count INTEGER,
                last_updated REAL
            )
        ''')
        
        # Policy recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policy_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_name TEXT,
                target_category TEXT,
                recommended_template TEXT,
                confidence_score REAL,
                expected_performance REAL,
                application_conditions TEXT,
                ai_reasoning TEXT,
                timestamp REAL
            )
        ''')
        
        # AI analysis sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_type TEXT,
                input_data TEXT,
                ai_analysis TEXT,
                insights_extracted TEXT,
                policies_created INTEGER,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    def query_ai_analyst(self, message: str, analysis_type: str = "general") -> str:
        """Query DeepSeek for analysis with specialized context"""
        
        system_contexts = {
            "pattern_analysis": """You are a DEEPSEEK AI PATTERN ANALYST specializing in optimization analysis.
Your expertise: Analyzing prompt optimization results to extract successful patterns and failure modes.
Your mission: Provide detailed insights into what makes prompts successful vs unsuccessful.
Your strength: Pattern recognition, statistical analysis, and strategic recommendations.""",
            
            "policy_creation": """You are a DEEPSEEK AI POLICY ARCHITECT specializing in reusable optimization strategies.
Your expertise: Creating actionable policies from successful patterns that can be applied to future optimizations.
Your mission: Transform analysis insights into concrete, reusable optimization templates.
Your strength: Strategic template design, conditional logic, and performance prediction.""",
            
            "insight_generation": """You are a DEEPSEEK AI INSIGHT SPECIALIST focusing on deep optimization understanding.
Your expertise: Extracting non-obvious insights from optimization data and predicting future performance.
Your mission: Discover hidden patterns and provide strategic guidance for optimization improvement.
Your strength: Deep analysis, trend identification, and predictive recommendations."""
        }
        
        context = system_contexts.get(analysis_type, system_contexts["pattern_analysis"])
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": message}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,  # More focused analysis
                "top_p": 0.9,
                "num_predict": 600,  # Longer analysis
                "stop": ["<think>", "</think>"],
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=90)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            # Clean response
            content = content.replace("<think>", "").replace("</think>", "")
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            
            return content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def analyze_optimization_results(self, results_data: Dict) -> Dict:
        """AI-powered analysis of optimization results"""
        
        print("\n🧠 AI ANALYST: Analyzing optimization results...")
        
        # Prepare data for AI analysis
        analysis_prompt = f"""DEEPSEEK AI PATTERN ANALYSIS TASK

OPTIMIZATION DATA TO ANALYZE:
{json.dumps(results_data, indent=2)}

ANALYSIS REQUIREMENTS:
1. PATTERN IDENTIFICATION: What patterns led to high scores vs low scores?
2. SUCCESS FACTORS: Which specific elements contributed to the best performance?
3. FAILURE MODES: What caused the worst performing attempts?
4. REPETITION ANALYSIS: Why did the AI generate repetitive prompts?
5. OPTIMIZATION OPPORTUNITIES: What could have been done differently?

SPECIFIC QUESTIONS TO ADDRESS:
- Why did the best score (0.877) occur in attempt 3?
- Why did identical prompts score differently across attempts?
- What premium descriptors were missing that could improve performance?
- How can we prevent AI repetition loops in future optimizations?
- What's the optimal prompt structure for this category?

PROVIDE DETAILED ANALYSIS IN THE FOLLOWING FORMAT:

PATTERN_ANALYSIS:
[Detailed analysis of successful vs failed patterns]

SUCCESS_FACTORS:
[Specific elements that drove high scores]

FAILURE_MODES:
[Root causes of poor performance]

REPETITION_CAUSES:
[Why AI got stuck in loops]

OPTIMIZATION_STRATEGY:
[Concrete recommendations for improvement]

POLICY_TEMPLATE:
[Reusable template for similar prompts]

Begin your expert analysis:"""

        ai_response = self.query_ai_analyst(analysis_prompt, "pattern_analysis")
        
        if "ERROR:" in ai_response:
            print(f"   ❌ AI Analysis failed: {ai_response}")
            return {"error": ai_response}
        
        print(f"   ✅ AI Analysis complete ({len(ai_response)} chars)")
        
        # Parse AI response into structured format
        analysis_sections = self.parse_ai_analysis(ai_response)
        
        # Store analysis session
        self.store_analysis_session("optimization_results", results_data, ai_response, analysis_sections)
        
        return {
            "ai_analysis": ai_response,
            "structured_analysis": analysis_sections,
            "timestamp": time.time()
        }

    def parse_ai_analysis(self, ai_response: str) -> Dict:
        """Parse AI analysis response into structured sections"""
        
        sections = {}
        current_section = None
        current_content = []
        
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            section_patterns = [
                "PATTERN_ANALYSIS:", "SUCCESS_FACTORS:", "FAILURE_MODES:",
                "REPETITION_CAUSES:", "OPTIMIZATION_STRATEGY:", "POLICY_TEMPLATE:"
            ]
            
            for pattern in section_patterns:
                if line.upper().startswith(pattern):
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = pattern.replace(':', '').lower()
                    current_content = []
                    break
            else:
                # Add content to current section
                if current_section and line:
                    current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def generate_optimization_policy(self, analysis_data: Dict, category: str) -> PolicyRecommendation:
        """AI-powered policy generation from analysis"""
        
        print("\n🎯 AI POLICY ARCHITECT: Creating optimization policy...")
        
        policy_prompt = f"""DEEPSEEK AI POLICY CREATION TASK

CATEGORY: {category}
ANALYSIS DATA: {json.dumps(analysis_data, indent=2)}

POLICY CREATION MISSION:
Create a concrete, reusable optimization policy that can be applied to future prompts in the '{category}' category.

POLICY REQUIREMENTS:
1. TEMPLATE: Specific prompt template with variable placeholders
2. CONDITIONS: When to apply this policy vs other approaches
3. PARAMETERS: Quality descriptors, structure elements, length guidelines
4. PERFORMANCE: Expected score range and success probability
5. FALLBACKS: Alternative approaches if primary template fails

PROVIDE POLICY IN THIS EXACT FORMAT:

POLICY_NAME: [Descriptive name for this policy]

TEMPLATE: [Exact prompt template with {{variable}} placeholders]

APPLICATION_CONDITIONS:
- [Condition 1: When to use this policy]
- [Condition 2: Category or prompt type requirements]
- [Condition 3: Performance threshold expectations]

EXPECTED_PERFORMANCE:
- Score Range: [X.XX - X.XX]
- Success Rate: [XX%]
- Ultra Achievement Probability: [XX%]

OPTIMIZATION_PARAMETERS:
- Required Elements: [List must-have components]
- Quality Descriptors: [Recommended premium language]
- Structure Guidelines: [Length, format, organization]

FAILURE_PREVENTION:
- [Specific measures to avoid common failure modes]
- [Anti-repetition strategies]
- [Quality validation checkpoints]

CONFIDENCE_ASSESSMENT: [1-10 scale with reasoning]

Begin policy creation:"""

        ai_response = self.query_ai_analyst(policy_prompt, "policy_creation")
        
        if "ERROR:" in ai_response:
            print(f"   ❌ Policy creation failed: {ai_response}")
            return None
        
        print(f"   ✅ Policy created ({len(ai_response)} chars)")
        
        # Parse policy response
        policy_data = self.parse_policy_response(ai_response, category)
        
        # Store policy
        self.store_policy_recommendation(policy_data)
        
        return policy_data

    def parse_policy_response(self, ai_response: str, category: str) -> PolicyRecommendation:
        """Parse AI policy response into structured policy"""
        
        # Extract policy components using regex
        policy_name = re.search(r'POLICY_NAME:\s*(.+)', ai_response, re.IGNORECASE)
        template = re.search(r'TEMPLATE:\s*(.+)', ai_response, re.IGNORECASE)
        confidence = re.search(r'CONFIDENCE_ASSESSMENT:\s*(\d+)', ai_response, re.IGNORECASE)
        
        # Extract application conditions
        conditions_match = re.search(r'APPLICATION_CONDITIONS:(.*?)(?=EXPECTED_PERFORMANCE|$)', ai_response, re.DOTALL | re.IGNORECASE)
        conditions = []
        if conditions_match:
            condition_lines = conditions_match.group(1).strip().split('\n')
            conditions = [line.strip('- ').strip() for line in condition_lines if line.strip().startswith('-')]
        
        # Extract expected performance
        performance_match = re.search(r'Score Range:\s*([0-9.-]+\s*-\s*[0-9.-]+)', ai_response, re.IGNORECASE)
        expected_score = 0.75  # Default
        if performance_match:
            score_range = performance_match.group(1)
            try:
                scores = [float(s.strip()) for s in score_range.split('-')]
                expected_score = statistics.mean(scores)
            except:
                pass
        
        return PolicyRecommendation(
            policy_name=policy_name.group(1).strip() if policy_name else f"Auto-Generated-{category}",
            target_category=category,
            recommended_template=template.group(1).strip() if template else "wbgmsst, {prompt}, white background",
            confidence_score=float(confidence.group(1)) / 10 if confidence else 0.7,
            expected_performance=expected_score,
            application_conditions=conditions,
            ai_reasoning=ai_response,
            timestamp=time.time()
        )

    def generate_strategic_insights(self, optimization_history: List[Dict]) -> Dict:
        """AI-powered strategic insights from optimization history"""
        
        print("\n💡 AI INSIGHT SPECIALIST: Generating strategic insights...")
        
        insight_prompt = f"""DEEPSEEK AI STRATEGIC INSIGHTS TASK

OPTIMIZATION HISTORY: {json.dumps(optimization_history, indent=2)}

STRATEGIC ANALYSIS MISSION:
Analyze the complete optimization history to extract deep insights and strategic recommendations for future ultra achievement.

INSIGHT CATEGORIES TO EXPLORE:
1. PERFORMANCE PATTERNS: What underlying patterns drive consistent high performance?
2. AI BEHAVIOR ANALYSIS: How can we improve AI decision-making for better results?
3. ULTRA ACHIEVEMENT STRATEGY: Specific path to consistent 0.96+ scores
4. LEARNING OPTIMIZATION: How to accelerate learning from each attempt
5. FUTURE PREDICTIONS: What optimizations will likely succeed/fail

PROVIDE INSIGHTS IN THIS FORMAT:

PERFORMANCE_INSIGHTS:
[Deep analysis of what drives high performance across attempts]

AI_OPTIMIZATION_INSIGHTS:
[How to improve AI behavior and decision-making]

ULTRA_ACHIEVEMENT_STRATEGY:
[Specific roadmap to consistent 0.96+ scores]

LEARNING_ACCELERATION:
[Methods to learn faster from each optimization attempt]

PREDICTIVE_RECOMMENDATIONS:
[What approaches will likely succeed in future optimizations]

BREAKTHROUGH_OPPORTUNITIES:
[Unexplored strategies that could lead to breakthrough performance]

Begin strategic insight generation:"""

        ai_response = self.query_ai_analyst(insight_prompt, "insight_generation")
        
        if "ERROR:" in ai_response:
            print(f"   ❌ Insight generation failed: {ai_response}")
            return {"error": ai_response}
        
        print(f"   ✅ Strategic insights generated ({len(ai_response)} chars)")
        
        # Parse insights
        insights = self.parse_ai_analysis(ai_response)
        
        return {
            "ai_insights": ai_response,
            "structured_insights": insights,
            "timestamp": time.time()
        }

    def store_analysis_session(self, session_type: str, input_data: Dict, ai_analysis: str, insights: Dict):
        """Store AI analysis session in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ai_analysis_sessions (session_type, input_data, ai_analysis, insights_extracted, policies_created, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_type,
            json.dumps(input_data),
            ai_analysis,
            json.dumps(insights),
            len(insights),
            time.time()
        ))
        
        conn.commit()
        conn.close()

    def store_policy_recommendation(self, policy: PolicyRecommendation):
        """Store policy recommendation in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO policy_recommendations (policy_name, target_category, recommended_template, confidence_score, expected_performance, application_conditions, ai_reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            policy.policy_name,
            policy.target_category,
            policy.recommended_template,
            policy.confidence_score,
            policy.expected_performance,
            json.dumps(policy.application_conditions),
            policy.ai_reasoning,
            policy.timestamp
        ))
        
        conn.commit()
        conn.close()

    def load_existing_policies(self, category: str = None) -> List[PolicyRecommendation]:
        """Load existing policies from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("SELECT * FROM policy_recommendations WHERE target_category = ? ORDER BY timestamp DESC", (category,))
        else:
            cursor.execute("SELECT * FROM policy_recommendations ORDER BY timestamp DESC")
        
        policies = []
        for row in cursor.fetchall():
            policy = PolicyRecommendation(
                policy_name=row[1],
                target_category=row[2],
                recommended_template=row[3],
                confidence_score=row[4],
                expected_performance=row[5],
                application_conditions=json.loads(row[6]) if row[6] else [],
                ai_reasoning=row[7],
                timestamp=row[8]
            )
            policies.append(policy)
        
        conn.close()
        return policies

    def analyze_and_create_policy(self, optimization_results: Dict) -> Dict:
        """Complete AI analysis and policy creation pipeline"""
        
        print("\n🚀 STARTING AI-POWERED ANALYSIS & POLICY CREATION")
        print("=" * 80)
        
        # Step 1: AI Analysis of results
        analysis_results = self.analyze_optimization_results(optimization_results)
        
        if "error" in analysis_results:
            return analysis_results
        
        # Step 2: Extract category and generate policy
        category = optimization_results.get("category", "technical")
        policy = self.generate_optimization_policy(analysis_results, category)
        
        # Step 3: Generate strategic insights
        insights = self.generate_strategic_insights([optimization_results])
        
        # Compile complete analysis
        complete_analysis = {
            "analysis": analysis_results,
            "policy": policy,
            "insights": insights,
            "timestamp": time.time()
        }
        
        print(f"\n✅ AI ANALYSIS & POLICY CREATION COMPLETE")
        print(f"   📊 Analysis: {len(analysis_results['ai_analysis'])} chars")
        print(f"   🎯 Policy: {policy.policy_name if policy else 'Failed'}")
        print(f"   💡 Insights: {len(insights['ai_insights'])} chars")
        
        return complete_analysis

def main():
    """Test the AI analyst with sample optimization results"""
    
    # Sample optimization results from the user's test
    sample_results = {
        "category": "technical",
        "target_prompt": "hexagonal prism steel structure",
        "ultra_target": 0.96,
        "attempts": 15,
        "best_score": 0.877,
        "average_score": 0.506,
        "score_range": {"min": -0.212, "max": 0.877},
        "ultra_achieved": False,
        "best_attempt": {
            "attempt_number": 3,
            "prompt": "wbgmsst, ultra-precision hexagonal prism steel structure, aerospace-grade quality, white background",
            "score": 0.877
        },
        "worst_attempt": {
            "attempt_number": 11,
            "prompt": "wbgmsst, ultra-precision hexagonal prism steel structure, aerospace-grade quality, white background",
            "score": -0.212
        },
        "key_issues": [
            "AI repetition loop - same prompt generated 12+ times",
            "Feedback resistance - ignored guidance to add more descriptors",
            "Score volatility - identical prompts scored vastly different"
        ],
        "performance_progression": [0.413, 0.797, 0.877, 0.638, 0.505, 0.294, 0.846, 0.433, 0.391, 0.784, -0.212, 0.547, 0.336, 0.229, 0.708]
    }
    
    print("🧠 TESTING AI ANALYST & POLICY MANAGER")
    print("=" * 80)
    
    # Create AI analyst
    analyst = DeepSeekAIAnalyst()
    
    # Run complete analysis and policy creation
    results = analyst.analyze_and_create_policy(sample_results)
    
    if "error" not in results:
        print(f"\n📋 ANALYSIS SUMMARY:")
        print(f"   🎯 Policy Created: {results['policy'].policy_name}")
        print(f"   📊 Confidence: {results['policy'].confidence_score:.1%}")
        print(f"   🏆 Expected Performance: {results['policy'].expected_performance:.3f}")
        print(f"   📝 Template: {results['policy'].recommended_template}")
        
        # Display key insights
        if 'structured_insights' in results['insights']:
            insights = results['insights']['structured_insights']
            if 'ultra_achievement_strategy' in insights:
                print(f"\n🎯 ULTRA ACHIEVEMENT STRATEGY:")
                print(f"   {insights['ultra_achievement_strategy'][:200]}...")
    
    print(f"\n✅ AI ANALYST TESTING COMPLETE")

if __name__ == "__main__":
    main() 