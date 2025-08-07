#!/usr/bin/env python3
"""
CLIP Alignment Analysis Summary
Analyzes CLIP alignment test results and generates detailed statistical summaries
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics

@dataclass
class TestResult:
    """Container for individual test results"""
    prompt1: str
    prompt2: str
    lora_endpoint: str
    p1_score: float
    p2_score: float
    improvement: float
    status: str
    test_type: str

@dataclass
class TestSession:
    """Container for a complete test session"""
    session_id: str
    base_prompt: str
    optimized_prompt: str
    results: List[TestResult]
    summary_stats: Dict[str, Any]

class CLIPAlignmentAnalyzer:
    """Analyzes CLIP alignment test results"""
    
    def __init__(self):
        self.test_sessions: List[TestSession] = []
        self.high_score_results: List[TestResult] = []  # Scores > 0.3
        self.all_results: List[TestResult] = []
        
    def parse_test_output(self, output_text: str) -> TestSession:
        """Parse test output and extract results"""
        lines = output_text.split('\n')
        
        # Extract prompts
        prompt1_match = re.search(r"Prompt 1: '([^']+)'", output_text)
        prompt2_match = re.search(r"Prompt 2: '([^']+)'", output_text)
        
        if not prompt1_match or not prompt2_match:
            raise ValueError("Could not extract prompts from output")
            
        prompt1 = prompt1_match.group(1)
        prompt2 = prompt2_match.group(1)
        
        # Extract results table
        results = []
        in_table = False
        
        for line in lines:
            if "LoRA Endpoint" in line and "P1 Score" in line:
                in_table = True
                continue
            elif in_table and line.strip() == "":
                in_table = False
                break
            elif in_table and "---" in line:
                continue
            elif in_table and line.strip():
                # Parse table row
                parts = line.split()
                if len(parts) >= 6:
                    lora_endpoint = parts[0]
                    try:
                        p1_score = float(parts[1])
                        p2_score = float(parts[2])
                        improvement = float(parts[3])
                        status = parts[5] if len(parts) > 5 else "UNKNOWN"
                        
                        result = TestResult(
                            prompt1=prompt1,
                            prompt2=prompt2,
                            lora_endpoint=lora_endpoint,
                            p1_score=p1_score,
                            p2_score=p2_score,
                            improvement=improvement,
                            status=status,
                            test_type="prompt_comparison"
                        )
                        results.append(result)
                        
                        # Track high scores
                        if p1_score > 0.3 or p2_score > 0.3:
                            self.high_score_results.append(result)
                        self.all_results.append(result)
                        
                    except ValueError:
                        continue
        
        # Extract summary statistics
        summary_stats = self._extract_summary_stats(output_text)
        
        session = TestSession(
            session_id=f"{prompt1[:30]}...",
            base_prompt=prompt1,
            optimized_prompt=prompt2,
            results=results,
            summary_stats=summary_stats
        )
        
        self.test_sessions.append(session)
        return session
    
    def _extract_summary_stats(self, output_text: str) -> Dict[str, Any]:
        """Extract summary statistics from output"""
        stats = {}
        
        # Extract key metrics
        patterns = {
            'total_endpoints': r'Total endpoints tested: (\d+)',
            'successful_generations': r'Successful generations: (\d+)',
            'failed_generations': r'Failed generations: (\d+)',
            'avg_p1_score': r'Average P1 score: ([\d.]+)',
            'avg_p2_score': r'Average P2 score: ([\d.]+)',
            'avg_improvement': r'Average improvement: ([+-]?[\d.]+)',
            'best_improvement': r'Best improvement: ([+-]?[\d.]+)',
            'worst_improvement': r'Worst improvement: ([+-]?[\d.]+)',
            'better_with_p2': r'Better with P2: (\d+)/(\d+)',
            'worse_with_p2': r'Worse with P2: (\d+)/(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output_text)
            if match:
                if key in ['better_with_p2', 'worse_with_p2']:
                    stats[key] = (int(match.group(1)), int(match.group(2)))
                else:
                    stats[key] = float(match.group(1))
        
        return stats
    
    def generate_high_score_analysis(self) -> str:
        """Generate analysis of results with scores > 0.3"""
        if not self.high_score_results:
            return "No results with scores > 0.3 found."
        
        # Group by score type
        p1_high_scores = [r for r in self.high_score_results if r.p1_score > 0.3]
        p2_high_scores = [r for r in self.high_score_results if r.p2_score > 0.3]
        
        analysis = []
        analysis.append("🎯 HIGH SCORE ANALYSIS (Scores > 0.3)")
        analysis.append("=" * 80)
        
        # P1 High Scores
        if p1_high_scores:
            analysis.append(f"\n📊 P1 HIGH SCORES ({len(p1_high_scores)} results):")
            analysis.append("-" * 60)
            analysis.append(f"{'LoRA Endpoint':<20} {'P1 Score':<12} {'P2 Score':<12} {'Improvement':<12} {'Status':<15}")
            analysis.append("-" * 60)
            
            for result in sorted(p1_high_scores, key=lambda x: x.p1_score, reverse=True):
                analysis.append(f"{result.lora_endpoint:<20} {result.p1_score:<12.4f} {result.p2_score:<12.4f} {result.improvement:<+12.4f} {result.status:<15}")
        
        # P2 High Scores
        if p2_high_scores:
            analysis.append(f"\n📊 P2 HIGH SCORES ({len(p2_high_scores)} results):")
            analysis.append("-" * 60)
            analysis.append(f"{'LoRA Endpoint':<20} {'P1 Score':<12} {'P2 Score':<12} {'Improvement':<12} {'Status':<15}")
            analysis.append("-" * 60)
            
            for result in sorted(p2_high_scores, key=lambda x: x.p2_score, reverse=True):
                analysis.append(f"{result.lora_endpoint:<20} {result.p1_score:<12.4f} {result.p2_score:<12.4f} {result.improvement:<+12.4f} {result.status:<15}")
        
        # Summary statistics
        if self.high_score_results:
            p1_scores = [r.p1_score for r in self.high_score_results if r.p1_score > 0.3]
            p2_scores = [r.p2_score for r in self.high_score_results if r.p2_score > 0.3]
            
            analysis.append(f"\n📈 HIGH SCORE STATISTICS:")
            if p1_scores:
                analysis.append(f"   P1 Scores > 0.3: {len(p1_scores)} results")
                analysis.append(f"   P1 Score Range: {min(p1_scores):.4f} - {max(p1_scores):.4f}")
                analysis.append(f"   P1 Average: {statistics.mean(p1_scores):.4f}")
            if p2_scores:
                analysis.append(f"   P2 Scores > 0.3: {len(p2_scores)} results")
                analysis.append(f"   P2 Score Range: {min(p2_scores):.4f} - {max(p2_scores):.4f}")
                analysis.append(f"   P2 Average: {statistics.mean(p2_scores):.4f}")
        
        return "\n".join(analysis)
    
    def generate_lora_performance_analysis(self) -> str:
        """Generate analysis of LoRA endpoint performance"""
        if not self.all_results:
            return "No results to analyze."
        
        # Group by LoRA endpoint
        lora_stats = defaultdict(lambda: {
            'count': 0,
            'p1_scores': [],
            'p2_scores': [],
            'improvements': [],
            'high_scores': 0
        })
        
        for result in self.all_results:
            stats = lora_stats[result.lora_endpoint]
            stats['count'] += 1
            stats['p1_scores'].append(result.p1_score)
            stats['p2_scores'].append(result.p2_score)
            stats['improvements'].append(result.improvement)
            if result.p1_score > 0.3 or result.p2_score > 0.3:
                stats['high_scores'] += 1
        
        analysis = []
        analysis.append("🎨 LoRA ENDPOINT PERFORMANCE ANALYSIS")
        analysis.append("=" * 100)
        analysis.append(f"{'LoRA Endpoint':<20} {'Tests':<8} {'Avg P1':<10} {'Avg P2':<10} {'Avg Imp':<10} {'High Scores':<12} {'Best P1':<10} {'Best P2':<10}")
        analysis.append("-" * 100)
        
        for lora, stats in sorted(lora_stats.items(), key=lambda x: statistics.mean(x[1]['p2_scores']), reverse=True):
            avg_p1 = statistics.mean(stats['p1_scores'])
            avg_p2 = statistics.mean(stats['p2_scores'])
            avg_imp = statistics.mean(stats['improvements'])
            best_p1 = max(stats['p1_scores'])
            best_p2 = max(stats['p2_scores'])
            
            analysis.append(f"{lora:<20} {stats['count']:<8} {avg_p1:<10.4f} {avg_p2:<10.4f} {avg_imp:<+10.4f} {stats['high_scores']:<12} {best_p1:<10.4f} {best_p2:<10.4f}")
        
        return "\n".join(analysis)
    
    def generate_prompt_effectiveness_analysis(self) -> str:
        """Generate analysis of prompt effectiveness"""
        if not self.test_sessions:
            return "No test sessions to analyze."
        
        analysis = []
        analysis.append("📝 PROMPT EFFECTIVENESS ANALYSIS")
        analysis.append("=" * 80)
        
        for i, session in enumerate(self.test_sessions, 1):
            analysis.append(f"\n🔍 SESSION {i}: {session.session_id}")
            analysis.append(f"   Base Prompt: '{session.base_prompt}'")
            analysis.append(f"   Optimized Prompt: '{session.optimized_prompt}'")
            
            if session.summary_stats:
                stats = session.summary_stats
                analysis.append(f"   Total Endpoints: {stats.get('total_endpoints', 'N/A')}")
                analysis.append(f"   Average P1 Score: {stats.get('avg_p1_score', 'N/A')}")
                analysis.append(f"   Average P2 Score: {stats.get('avg_p2_score', 'N/A')}")
                analysis.append(f"   Average Improvement: {stats.get('avg_improvement', 'N/A')}")
                
                if 'better_with_p2' in stats:
                    better, total = stats['better_with_p2']
                    analysis.append(f"   Better with P2: {better}/{total} ({better/total*100:.1f}%)")
        
        return "\n".join(analysis)
    
    def generate_comprehensive_summary(self) -> str:
        """Generate comprehensive analysis summary"""
        if not self.all_results:
            return "No results to analyze."
        
        summary = []
        summary.append("🎯 COMPREHENSIVE CLIP ALIGNMENT ANALYSIS")
        summary.append("=" * 100)
        
        # Overall statistics
        all_p1_scores = [r.p1_score for r in self.all_results]
        all_p2_scores = [r.p2_score for r in self.all_results]
        all_improvements = [r.improvement for r in self.all_results]
        
        summary.append(f"\n📊 OVERALL STATISTICS:")
        summary.append(f"   Total Results: {len(self.all_results)}")
        summary.append(f"   Test Sessions: {len(self.test_sessions)}")
        summary.append(f"   High Scores (>0.3): {len(self.high_score_results)}")
        summary.append(f"   Average P1 Score: {statistics.mean(all_p1_scores):.4f}")
        summary.append(f"   Average P2 Score: {statistics.mean(all_p2_scores):.4f}")
        summary.append(f"   Average Improvement: {statistics.mean(all_improvements):+.4f}")
        summary.append(f"   Best P1 Score: {max(all_p1_scores):.4f}")
        summary.append(f"   Best P2 Score: {max(all_p2_scores):.4f}")
        summary.append(f"   Best Improvement: {max(all_improvements):+.4f}")
        
        # Score distribution
        p1_high = sum(1 for s in all_p1_scores if s > 0.3)
        p2_high = sum(1 for s in all_p2_scores if s > 0.3)
        summary.append(f"\n📈 SCORE DISTRIBUTION:")
        summary.append(f"   P1 Scores > 0.3: {p1_high}/{len(all_p1_scores)} ({p1_high/len(all_p1_scores)*100:.1f}%)")
        summary.append(f"   P2 Scores > 0.3: {p2_high}/{len(all_p2_scores)} ({p2_high/len(all_p2_scores)*100:.1f}%)")
        
        # Top performers
        top_p1 = sorted(self.all_results, key=lambda x: x.p1_score, reverse=True)[:5]
        top_p2 = sorted(self.all_results, key=lambda x: x.p2_score, reverse=True)[:5]
        top_improvements = sorted(self.all_results, key=lambda x: x.improvement, reverse=True)[:5]
        
        summary.append(f"\n🏆 TOP 5 P1 SCORES:")
        for i, result in enumerate(top_p1, 1):
            summary.append(f"   {i}. {result.lora_endpoint}: {result.p1_score:.4f} (P2: {result.p2_score:.4f}, Imp: {result.improvement:+.4f})")
        
        summary.append(f"\n🏆 TOP 5 P2 SCORES:")
        for i, result in enumerate(top_p2, 1):
            summary.append(f"   {i}. {result.lora_endpoint}: {result.p2_score:.4f} (P1: {result.p1_score:.4f}, Imp: {result.improvement:+.4f})")
        
        summary.append(f"\n🏆 TOP 5 IMPROVEMENTS:")
        for i, result in enumerate(top_improvements, 1):
            summary.append(f"   {i}. {result.lora_endpoint}: {result.improvement:+.4f} (P1: {result.p1_score:.4f} → P2: {result.p2_score:.4f})")
        
        return "\n".join(summary)

def main():
    """Main function to demonstrate the analyzer"""
    analyzer = CLIPAlignmentAnalyzer()
    
    # Example usage - you would parse your actual test outputs here
    print("CLIP Alignment Analysis Tool")
    print("=" * 50)
    print("This tool analyzes CLIP alignment test results and generates detailed summaries.")
    print("To use with your data, parse the test outputs and call the analysis methods.")
    
    # Example of how to use the analyzer
    print("\nExample usage:")
    print("1. Create analyzer instance")
    print("2. Parse test outputs with parse_test_output()")
    print("3. Generate analyses with the various analysis methods")
    print("4. Focus on high scores (>0.3) for detailed examination")

if __name__ == "__main__":
    main() 