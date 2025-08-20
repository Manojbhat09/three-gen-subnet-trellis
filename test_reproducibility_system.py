#!/usr/bin/env python3
"""
Comprehensive Tests for LLM Close Prompt Reproducibility System

This test file demonstrates how the reproducibility system works step by step,
showing the jump from "found close gold prompt" to "reproducibility optimization SUCCESS".
"""

import unittest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from difflib import SequenceMatcher

# Import the reproducibility system
from llm_close_prompt_reproducibility_test import (
    LLMClosePromptReproducibility,
    calculate_similarity,
    extract_true_prompt
)

class TestReproducibilitySystem(unittest.TestCase):
    """Test suite for the reproducibility system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary episodic memory file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.episodic_memory_file = os.path.join(self.temp_dir, "test_episodic_memory.json")
        
        # Create sample episodic memory data
        self.sample_episodic_data = {
            "optimization_sessions": [
                {
                    "original_prompt": "intricate sandstone sculpture of cat lounging",
                    "attempts": [
                        {
                            "optimized_prompt": "cylindrical glass of bubbly lemonade",
                            "validation_score": 0.7608
                        },
                        {
                            "optimized_prompt": "rough stone cat statue",
                            "validation_score": 0.4500
                        }
                    ]
                },
                {
                    "original_prompt": "shiny metal robot with glowing eyes",
                    "attempts": [
                        {
                            "optimized_prompt": "metallic android with luminous optics",
                            "validation_score": 0.8900
                        }
                    ]
                },
                {
                    "original_prompt": "wooden table with carved details",
                    "attempts": [
                        {
                            "optimized_prompt": "carved wooden table with intricate patterns",
                            "validation_score": 0.7200
                        }
                    ]
                }
            ]
        }
        
        # Write sample data to file
        with open(self.episodic_memory_file, 'w') as f:
            json.dump(self.sample_episodic_data, f)
        
        # Mock the LLM optimizer to avoid actual API calls
        self.mock_optimizer = Mock()
        self.mock_optimizer._query_llm = Mock(return_value='{"core_subject": "test", "enhancements": {"quality_adjectives": [], "material_details": [], "light_interaction": [], "context": []}}')
        
        # Create reproducibility system with mocked components
        with patch('llm_close_prompt_reproducibility_test.LLMPromptOptimizer', return_value=self.mock_optimizer):
            self.repro_system = LLMClosePromptReproducibility(
                episodic_memory_file=self.episodic_memory_file,
                use_vllm=False,
                ollama_url="http://localhost:11434"
            )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_extract_true_prompt(self):
        """Test extracting true prompt from modified strings"""
        # Test normal prompt
        self.assertEqual(
            extract_true_prompt("a simple prompt"),
            "a simple prompt"
        )
        
        # Test prompt with "Original Prompt:" suffix
        self.assertEqual(
            extract_true_prompt("some text Original Prompt: the real prompt"),
            "the real prompt"
        )
        
        # Test prompt with multiple "Original Prompt:" occurrences
        self.assertEqual(
            extract_true_prompt("text Original Prompt: first Original Prompt: final"),
            "final"
        )
    
    def test_calculate_similarity(self):
        """Test similarity calculation between prompts"""
        # Test identical prompts
        self.assertEqual(calculate_similarity("hello world", "hello world"), 1.0)
        
        # Test similar prompts
        sim = calculate_similarity("hello world", "hello world!")
        self.assertGreater(sim, 0.8)
        
        # Test different prompts
        sim = calculate_similarity("hello world", "goodbye universe")
        self.assertLess(sim, 0.5)
        
        # Test empty prompts
        self.assertEqual(calculate_similarity("", ""), 1.0)
        self.assertEqual(calculate_similarity("hello", ""), 0.0)
    
    def test_load_episodic_memory(self):
        """Test loading episodic memory from file"""
        # Test successful loading
        gold_results = self.repro_system._load_episodic_memory()
        
        self.assertIn("intricate sandstone sculpture of cat lounging", gold_results)
        self.assertIn("shiny metal robot with glowing eyes", gold_results)
        self.assertIn("wooden table with carved details", gold_results)
        
        # Test that best attempts are selected
        cat_data = gold_results["intricate sandstone sculpture of cat lounging"]
        self.assertIn("method_2_hybrid_example", cat_data)
        self.assertEqual(
            cat_data["method_2_hybrid_example"]["optimized_prompt"],
            "cylindrical glass of bubbly lemonade"
        )
        self.assertEqual(
            cat_data["method_2_hybrid_example"]["validation_results"]["validation_engine_score"],
            0.7608
        )
    
    def test_find_closest_gold_prompt(self):
        """Test finding the closest gold prompt to a given prompt"""
        # Test finding close prompt - use a slightly different prompt to get partial similarity
        result = self.repro_system.find_closest_gold_prompt(
            "sandstone cat sculpture", 
            min_similarity=0.3
        )
        
        self.assertIsNotNone(result)
        gold_prompt, gold_score, similarity = result
        
        # Should find the cat-related prompt (even though it's an exact match in our test data)
        # The test data has "intricate sandstone sculpture of cat lounging" which should match
        self.assertGreater(similarity, 0.3)
        self.assertEqual(gold_score, 0.7608)
        
        # Test with no close match
        result = self.repro_system.find_closest_gold_prompt(
            "completely unrelated prompt about quantum physics",
            min_similarity=0.8
        )
        self.assertIsNone(result)
    
    def test_extract_patterns(self):
        """Test extracting patterns from prompts using LLM"""
        # Mock the LLM response for pattern extraction
        mock_response = '''
        {
            "core_subject": "sandstone cat sculpture",
            "enhancements": {
                "quality_adjectives": ["intricate"],
                "material_details": ["sandstone"],
                "light_interaction": [],
                "context": []
            }
        }
        '''
        self.mock_optimizer._query_llm.return_value = mock_response
        
        # Test pattern extraction
        patterns = self.repro_system.extract_patterns("intricate sandstone sculpture of cat lounging")
        
        self.assertIn("core_subject", patterns)
        self.assertIn("enhancements", patterns)
        self.assertEqual(patterns["core_subject"], "sandstone cat sculpture")
        self.assertIn("intricate", patterns["enhancements"]["quality_adjectives"])
        self.assertIn("sandstone", patterns["enhancements"]["material_details"])
    
    def test_merge_components_intelligently(self):
        """Test intelligent merging of prompt components"""
        original_components = {
            "core_subject": "stone cat",
            "enhancements": {
                "quality_adjectives": ["rough"],
                "material_details": ["stone"],
                "light_interaction": [],
                "context": []
            }
        }
        
        gold_components = {
            "core_subject": "cylindrical glass",
            "enhancements": {
                "quality_adjectives": ["bubbly"],
                "material_details": ["glass"],
                "light_interaction": ["sparkling"],
                "context": []
            }
        }
        
        # Test merging with high similarity (should preserve original core subject)
        merged = self.repro_system.merge_components_intelligently(
            original_components, gold_components, similarity=0.8
        )
        
        # Should preserve original core subject for all similarity levels
        self.assertEqual(merged["core_subject"], "stone cat")
        self.assertIn("bubbly", merged["enhancements"]["quality_adjectives"])
        self.assertIn("sparkling", merged["enhancements"]["light_interaction"])
        
        # Test merging with low similarity (should also preserve original core subject)
        merged = self.repro_system.merge_components_intelligently(
            original_components, gold_components, similarity=0.4
        )
        
        # Should preserve original core subject
        self.assertEqual(merged["core_subject"], "stone cat")
        # But should still include gold enhancements
        self.assertIn("sparkling", merged["enhancements"]["light_interaction"])
    
    def test_reconstruct_prompt(self):
        """Test reconstructing prompt from merged components"""
        # Mock the LLM response for prompt reconstruction
        mock_response = "cylindrical glass of bubbly lemonade, white background"
        self.mock_optimizer._query_llm.return_value = mock_response
        
        components = {
            "core_subject": "cylindrical glass",
            "enhancements": {
                "quality_adjectives": ["bubbly"],
                "material_details": ["glass"],
                "light_interaction": [],
                "context": []
            }
        }
        
        # Test prompt reconstruction
        reconstructed = self.repro_system.reconstruct_prompt(components)
        
        self.assertEqual(reconstructed, "cylindrical glass of bubbly lemonade, white background")
        
        # Verify LLM was called with correct system prompt
        call_args = self.mock_optimizer._query_llm.call_args
        self.assertIn("cylindrical glass", call_args[0][0])  # system prompt
        self.assertEqual(call_args[0][1], "")  # user prompt (empty)
    
    def test_optimize_prompt_with_reproducibility_full_flow(self):
        """Test the complete reproducibility optimization flow"""
        # This test shows the complete flow from finding a gold prompt to optimization success
        
        # Mock the LLM responses for the entire flow
        self.mock_optimizer._query_llm.side_effect = [
            # Response for extracting patterns from original prompt
            '''
            {
                "core_subject": "sandstone cat sculpture",
                "enhancements": {
                    "quality_adjectives": ["intricate"],
                    "material_details": ["sandstone"],
                    "light_interaction": [],
                    "context": []
                }
            }
            ''',
            # Response for extracting patterns from gold prompt
            '''
            {
                "core_subject": "cylindrical glass",
                "enhancements": {
                    "quality_adjectives": ["bubbly"],
                    "material_details": ["glass"],
                    "light_interaction": [],
                    "context": []
                }
            }
            ''',
            # Response for reconstructing final prompt
            "cylindrical glass of bubbly lemonade, white background"
        ]
        
        # Test the complete optimization
        result = self.repro_system.optimize_prompt_with_reproducibility(
            "intricate sandstone sculpture of cat lounging",
            min_similarity=0.3,
            run_validation=False  # Skip validation for testing
        )
        
        # Verify the result structure
        self.assertIsNotNone(result)
        self.assertIn("original_prompt", result)
        self.assertIn("optimized_prompt", result)
        self.assertIn("gold_prompt", result)
        self.assertIn("similarity", result)
        self.assertIn("gold_score", result)
        
        # Verify the optimization worked
        self.assertEqual(
            result["original_prompt"], 
            "intricate sandstone sculpture of cat lounging"
        )
        # The optimized prompt should preserve the original intent while adding enhancements
        self.assertIn("sandstone cat sculpture", result["optimized_prompt"])
        self.assertIn("intricate", result["optimized_prompt"])
        self.assertIn("bubbly", result["optimized_prompt"])
        self.assertIn("glass", result["optimized_prompt"])
        self.assertIn("white background", result["optimized_prompt"])
        self.assertEqual(result["gold_score"], 0.7608)
        self.assertGreater(result["similarity"], 0.3)
        
        # Verify the optimization method
        self.assertEqual(result["optimization_method"], "reproducibility_merge")
        
        print("\n🎯 COMPLETE OPTIMIZATION FLOW TEST:")
        print(f"   Original: '{result['original_prompt']}'")
        print(f"   Found gold prompt: '{result['gold_prompt']}' (score: {result['gold_score']:.4f})")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Optimized: '{result['optimized_prompt']}'")
        print(f"   Method: {result['optimization_method']}")
    
    def test_optimize_prompt_with_reproducibility_no_match(self):
        """Test optimization when no close gold prompt is found"""
        # Test with a prompt that has no close match
        result = self.repro_system.optimize_prompt_with_reproducibility(
            "completely unrelated prompt about quantum physics",
            min_similarity=0.8,  # Very high threshold
            run_validation=False
        )
        
        # Should return None when no match found
        self.assertIsNone(result)
    
    def test_optimize_prompt_with_reproducibility_with_validation(self):
        """Test optimization with validation enabled"""
        # Mock the validator to return a score
        with patch.object(self.repro_system, '_run_validator') as mock_validator:
            mock_validator.return_value = {
                "validation_engine_score": 0.7500
            }
            
            # Mock LLM responses
            self.mock_optimizer._query_llm.side_effect = [
                # Pattern extraction responses
                '{"core_subject": "test", "enhancements": {"quality_adjectives": [], "material_details": [], "light_interaction": [], "context": []}}',
                '{"core_subject": "test", "enhancements": {"quality_adjectives": [], "material_details": [], "light_interaction": [], "context": []}}',
                # Prompt reconstruction
                "test optimized prompt, white background"
            ]
            
            # Test with validation
            result = self.repro_system.optimize_prompt_with_reproducibility(
                "intricate sandstone sculpture of cat lounging",
                min_similarity=0.3,
                run_validation=True
            )
            
            # Verify validation was called
            mock_validator.assert_called_once()
            
            # Verify validation results are included
            self.assertIn("optimized_score", result)
            self.assertIn("score_improvement", result)
            self.assertIn("improvement_percentage", result)
            self.assertIn("status", result)
    
    def test_items_conflict_detection(self):
        """Test detection of conflicting enhancement items"""
        # Test conflicting items
        self.assertTrue(self.repro_system._items_conflict("bright", "dark"))
        self.assertTrue(self.repro_system._items_conflict("shiny", "matte"))
        self.assertTrue(self.repro_system._items_conflict("smooth", "rough"))
        
        # Test non-conflicting items
        self.assertFalse(self.repro_system._items_conflict("bright", "shiny"))
        self.assertFalse(self.repro_system._items_conflict("smooth", "soft"))
        self.assertFalse(self.repro_system._items_conflict("large", "heavy"))
    
    def test_extract_clean_json(self):
        """Test robust JSON extraction from LLM responses"""
        # Test clean JSON
        clean_json = '{"key": "value"}'
        result = self.repro_system._extract_clean_json(clean_json)
        self.assertEqual(result, {"key": "value"})
        
        # Test JSON with extra text
        messy_json = 'Here is the result: {"key": "value"} and some extra text'
        result = self.repro_system._extract_clean_json(messy_json)
        self.assertEqual(result, {"key": "value"})
        
        # Test JSON with unquoted properties (should be fixed)
        unquoted_json = '{key: "value", number: 42}'
        result = self.repro_system._extract_clean_json(unquoted_json)
        self.assertIn("key", result)
        self.assertEqual(result["key"], "value")
    
    def test_similarity_calculation_edge_cases(self):
        """Test similarity calculation with edge cases"""
        # Test very short prompts
        sim = calculate_similarity("a", "b")
        self.assertLess(sim, 1.0)
        
        # Test prompts with special characters
        sim = calculate_similarity("hello-world!", "hello world")
        self.assertGreater(sim, 0.5)
        
        # Test prompts with numbers
        sim = calculate_similarity("item 123", "item 456")
        self.assertGreater(sim, 0.3)
        
        # Test case sensitivity
        sim1 = calculate_similarity("Hello World", "hello world")
        sim2 = calculate_similarity("HELLO WORLD", "hello world")
        self.assertAlmostEqual(sim1, sim2, places=3)


class TestReproducibilitySystemIntegration(unittest.TestCase):
    """Integration tests showing how the system works in practice"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create a more realistic episodic memory
        self.temp_dir = tempfile.mkdtemp()
        self.episodic_memory_file = os.path.join(self.temp_dir, "integration_test_memory.json")
        
        # Create realistic episodic memory with multiple sessions
        self.realistic_episodic_data = [
            {
                "session_id": "session_001",
                "optimization_sessions": [
                    {
                        "original_prompt": "intricate sandstone sculpture of cat lounging",
                        "attempts": [
                            {
                                "optimized_prompt": "cylindrical glass of bubbly lemonade",
                                "validation_score": 0.7608
                            },
                            {
                                "optimized_prompt": "rough stone cat statue",
                                "validation_score": 0.4500
                            }
                        ]
                    }
                ]
            },
            {
                "session_id": "session_002", 
                "optimization_sessions": [
                    {
                        "original_prompt": "shiny metal robot with glowing eyes",
                        "attempts": [
                            {
                                "optimized_prompt": "metallic android with luminous optics",
                                "validation_score": 0.8900
                            }
                        ]
                    }
                ]
            }
        ]
        
        with open(self.episodic_memory_file, 'w') as f:
            json.dump(self.realistic_episodic_data, f)
        
        # Mock optimizer
        self.mock_optimizer = Mock()
        self.mock_optimizer._query_llm = Mock()
        
        with patch('llm_close_prompt_reproducibility_test.LLMPromptOptimizer', return_value=self.mock_optimizer):
            self.repro_system = LLMClosePromptReproducibility(
                episodic_memory_file=self.episodic_memory_file,
                use_vllm=False,
                ollama_url="http://localhost:11434"
            )
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_new_format_episodic_memory_loading(self):
        """Test loading episodic memory in the new multi-session format"""
        # Verify the new format is loaded correctly
        gold_results = self.repro_system._load_episodic_memory()
        
        # Should have prompts from both sessions
        self.assertIn("intricate sandstone sculpture of cat lounging", gold_results)
        self.assertIn("shiny metal robot with glowing eyes", gold_results)
        
        # Verify best scores are selected
        cat_score = gold_results["intricate sandstone sculpture of cat lounging"]["method_2_hybrid_example"]["validation_results"]["validation_engine_score"]
        robot_score = gold_results["shiny metal robot with glowing eyes"]["method_2_hybrid_example"]["validation_results"]["validation_engine_score"]
        
        self.assertEqual(cat_score, 0.7608)  # Best of the two attempts
        self.assertEqual(robot_score, 0.8900)  # Only attempt
    
    def test_complete_optimization_workflow(self):
        """Test the complete optimization workflow with realistic data"""
        # Mock LLM responses for the complete workflow
        self.mock_optimizer._query_llm.side_effect = [
            # Extract patterns from original prompt
            '{"core_subject": "sandstone cat sculpture", "enhancements": {"quality_adjectives": ["intricate"], "material_details": ["sandstone"], "light_interaction": [], "context": []}}',
            # Extract patterns from gold prompt  
            '{"core_subject": "cylindrical glass", "enhancements": {"quality_adjectives": ["bubbly"], "material_details": ["glass"], "light_interaction": [], "context": []}}',
            # Reconstruct final prompt
            "cylindrical glass of bubbly lemonade, white background"
        ]
        
        # Run the complete optimization
        result = self.repro_system.optimize_prompt_with_reproducibility(
            "intricate sandstone sculpture of cat lounging",
            min_similarity=0.3,
            run_validation=False
        )
        
        # Verify the complete result
        self.assertIsNotNone(result)
        self.assertEqual(result["original_prompt"], "intricate sandstone sculpture of cat lounging")
        # The optimized prompt should preserve the original intent while adding enhancements
        self.assertIn("sandstone cat sculpture", result["optimized_prompt"])
        self.assertIn("intricate", result["optimized_prompt"])
        self.assertIn("bubbly", result["optimized_prompt"])
        self.assertIn("glass", result["optimized_prompt"])
        self.assertIn("white background", result["optimized_prompt"])
        self.assertEqual(result["gold_prompt"], "cylindrical glass of bubbly lemonade")
        self.assertEqual(result["gold_score"], 0.7608)
        self.assertGreater(result["similarity"], 0.3)
        
        print("\n🚀 COMPLETE INTEGRATION WORKFLOW TEST:")
        print(f"   Input: '{result['original_prompt']}'")
        print(f"   Found gold: '{result['gold_prompt']}' (score: {result['gold_score']:.4f})")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Output: '{result['optimized_prompt']}'")
        print(f"   Method: {result['optimization_method']}")
        print(f"   Status: {result['status']}")


def run_demo():
    """Run a demonstration of how the reproducibility system works"""
    print("🎯 REPRODUCIBILITY SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create a temporary test file
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "demo_memory.json")
    
    # Create demo episodic memory
    demo_data = {
        "optimization_sessions": [
            {
                "original_prompt": "intricate sandstone sculpture of cat lounging",
                "attempts": [
                    {
                        "optimized_prompt": "cylindrical glass of bubbly lemonade",
                        "validation_score": 0.7608
                    }
                ]
            }
        ]
    }
    
    with open(test_file, 'w') as f:
        json.dump(demo_data, f)
    
    try:
        # Create reproducibility system
        with patch('llm_close_prompt_reproducibility_test.LLMPromptOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer._query_llm = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            # Mock the LLM responses
            mock_optimizer._query_llm.side_effect = [
                # Extract patterns from original
                '{"core_subject": "sandstone cat sculpture", "enhancements": {"quality_adjectives": ["intricate"], "material_details": ["sandstone"], "light_interaction": [], "context": []}}',
                # Extract patterns from gold
                '{"core_subject": "cylindrical glass", "enhancements": {"quality_adjectives": ["bubbly"], "material_details": ["glass"], "light_interaction": [], "context": []}}',
                # Reconstruct prompt
                "cylindrical glass of bubbly lemonade, white background"
            ]
            
            repro_system = LLMClosePromptReproducibility(
                episodic_memory_file=test_file,
                use_vllm=False,
                ollama_url="http://localhost:11434"
            )
            
            print("📚 Step 1: Loading episodic memory...")
            gold_count = len(repro_system.gold_standard_results)
            print(f"   Loaded {gold_count} gold prompts")
            
            print("\n🔍 Step 2: Finding close gold prompt...")
            original_prompt = "intricate sandstone sculpture of cat lounging"
            closest_match = repro_system.find_closest_gold_prompt(original_prompt, min_similarity=0.3)
            
            if closest_match:
                gold_prompt, gold_score, similarity = closest_match
                print(f"   ✅ Found close gold prompt!")
                print(f"   Original: '{original_prompt}'")
                print(f"   Gold: '{gold_prompt}'")
                print(f"   Score: {gold_score:.4f}")
                print(f"   Similarity: {similarity:.3f}")
                
                print("\n🔄 Step 3: Running complete optimization...")
                result = repro_system.optimize_prompt_with_reproducibility(
                    original_prompt, min_similarity=0.3, run_validation=False
                )
                
                if result:
                    print(f"   🎯 REPRODUCIBILITY OPTIMIZATION SUCCESS!")
                    print(f"   Original: '{result['original_prompt']}'")
                    print(f"   Optimized: '{result['optimized_prompt']}'")
                    print(f"   Similarity: {result['similarity']:.3f}")
                    print(f"   Gold score: {result['gold_score']:.4f}")
                    print(f"   Method: {result['optimization_method']}")
                    print(f"   Status: {result['status']}")
                    
                    print("\n💡 This demonstrates how the system:")
                    print("   1. Finds a similar gold prompt in episodic memory")
                    print("   2. Extracts patterns from both prompts using LLM")
                    print("   3. Intelligently merges components")
                    print("   4. Reconstructs an optimized prompt")
                    print("   5. Returns the complete optimization result")
                else:
                    print("   ❌ Optimization failed")
            else:
                print("   ❌ No close gold prompt found")
                
    finally:
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("🧪 Running Reproducibility System Tests...")
    print("=" * 60)
    
    # Run the demo first
    run_demo()
    
    print("\n" + "=" * 60)
    print("🧪 Running Unit Tests...")
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\n💡 Key Insights:")
    print("   • The 'jump' from 'found close gold prompt' to 'SUCCESS' happens")
    print("     because optimize_prompt_with_reproducibility() does 4 steps:")
    print("     1. Find closest gold prompt")
    print("     2. Extract patterns from both prompts")
    print("     3. Merge components intelligently") 
    print("     4. Reconstruct optimized prompt")
    print("   • Each step uses LLM calls for pattern extraction and reconstruction")
    print("   • The system combines episodic memory knowledge with LLM intelligence")
    print("   • This provides fast, deterministic optimization without full RL loops")
