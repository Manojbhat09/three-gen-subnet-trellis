#!/usr/bin/env python3
"""
LLM Close Prompt Reproducibility Test

This script implements reproducibility testing as a pre-optimization layer for the continuous orchestrator.
It finds the closest gold prompt to the original prompt and intelligently merges their components to create
a high-scoring prompt without needing full optimization.

The process is as follows:
1. Loads episodic memory to find gold standard prompts
2. For a given original prompt, finds the closest gold prompt by semantic similarity
3. Extracts components from both the original and gold prompts
4. Intelligently merges components, keeping high-scoring elements from the gold prompt
5. Reconstructs a new prompt with the merged components
6. Returns the optimized prompt for use in the continuous orchestrator

This provides a fast, deterministic way to improve prompts without full optimization.
"""
import argparse
import json
import os
import sys
import re
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

# Import episodic test prompts
from episodic_test_prompts import EPISODIC_TEST_PROMPTS

# Import the vLLM-enabled version of LLMPromptOptimizer
from llm_prompt_optimizer_v12_f1_lora import LLMPromptOptimizer

def extract_true_prompt(original_prompt: str) -> str:
    """Extract the true prompt from a potentially modified original prompt string."""
    # Try to find "Original Prompt: ..." at the end
    if "Original Prompt:" in original_prompt:
        return original_prompt.split("Original Prompt:")[-1].strip()
    return original_prompt

def calculate_similarity(prompt1: str, prompt2: str) -> float:
    """Calculate similarity between two prompts using sequence matching."""
    return SequenceMatcher(None, prompt1.lower(), prompt2.lower()).ratio()

class LLMClosePromptReproducibility:
    """Manages the close prompt reproducibility system."""

    def __init__(self, episodic_memory_file: str = "episodic_logs/episodic_memory.json", 
                 use_vllm: bool = True, vllm_url: str = "http://localhost:9000", 
                 vllm_model: str = "llama-3-2-3b-it", ollama_url: str = "http://localhost:11434", ollama_model: str = "llama3.2:3b"):
        self.episodic_memory_file = episodic_memory_file
        self.gold_standard_results = self._load_episodic_memory()
        
        # Initialize optimizer with appropriate LLM provider
        if use_vllm:
            self.optimizer = LLMPromptOptimizer(
                ollama_url=ollama_url,
                model=ollama_model,  # Default model for compatibility
                use_vllm=True,
                vllm_url=vllm_url,
                vllm_model=vllm_model
            )
        else:
            self.optimizer = LLMPromptOptimizer(ollama_url=ollama_url)
        
        # Cache for similarity calculations
        self.similarity_cache = {}

    def _run_validator(self, original_prompt: str, optimized_prompt: str = None, endpoint: str = "generate/") -> Dict[str, Any]:
        """Runs the subnet_accurate_validator.py script to get ground truth score."""
        if optimized_prompt and optimized_prompt != original_prompt:
            print(f"  🔍 Validating optimized prompt: '{optimized_prompt[:60]}...'")
            print(f"  🎯 Computing scores against original: '{original_prompt[:60]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\" --endpoint \"{endpoint}\""
            ]
        else:
            print(f"  🔍 Validating prompt: '{original_prompt[:60]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" --endpoint \"{endpoint}\""
            ]
            try:
                process = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
                validator_output_file = "subnet_validation_results.json"
                if os.path.exists(validator_output_file):
                    with open(validator_output_file, 'r') as f:
                        return json.load(f)
                return {"error": "subnet_validation_results.json not found."}
            except Exception as e:
                print(f"  ❌ Validation script failed for prompt: '{original_prompt[:60]}...'")
                return {"error": f"Validator script failed: {e}"}

    def _load_episodic_memory(self) -> Dict[str, Any]:
        """Loads episodic memory and extracts gold standard results."""
        if not os.path.exists(self.episodic_memory_file):
            print(f"❌ Error: Episodic memory file not found at '{self.episodic_memory_file}'.")
            return {}
        
        try:
            with open(self.episodic_memory_file, 'r') as f:
                episodic_data = json.load(f)
            
            # Handle both old format (dict) and new format (list of sessions)
            if isinstance(episodic_data, dict):
                # Old format - single session
                print(f"📚 Loading from old single-session memory format")
                optimization_sessions = episodic_data.get("optimization_sessions", [])
            elif isinstance(episodic_data, list):
                # New format - multiple sessions
                print(f"📚 Loading from new multi-session memory format ({len(episodic_data)} sessions)")
                # Combine optimization sessions from all sessions
                optimization_sessions = []
                for session_data in episodic_data:
                    if isinstance(session_data, dict):
                        session_sessions = session_data.get("optimization_sessions", [])
                        optimization_sessions.extend(session_sessions)
                        print(f"   📖 Session {session_data.get('session_id', 'unknown')}: {len(session_sessions)} optimization sessions")
            else:
                print(f"❌ Error: Unknown episodic memory format: {type(episodic_data)}")
                return {}
            
            # Convert to gold standard format
            gold_standard_results = {}
            
            for session in optimization_sessions:
                original_prompt = extract_true_prompt(session.get("original_prompt", ""))
                if not original_prompt:
                    continue
                
                # Find the best attempt in this session
                best_attempt = None
                best_score = 0.0
                
                for attempt in session.get("attempts", []):
                    validation_score = attempt.get("validation_score")
                    if validation_score is not None and validation_score > best_score:
                        best_score = validation_score
                        best_attempt = attempt
                
                if best_attempt:
                    # Store in format similar to experiment_results.json
                    if original_prompt not in gold_standard_results:
                        gold_standard_results[original_prompt] = {}
                    
                    # Use method_2_hybrid_example format for consistency
                    gold_standard_results[original_prompt]["method_2_hybrid_example"] = {
                        "optimized_prompt": best_attempt["optimized_prompt"],
                        "validation_results": {
                            "validation_engine_score": best_score
                        }
                    }
            
            print(f"✅ Loaded {len(gold_standard_results)} gold standard prompts from episodic memory")
            return gold_standard_results
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error parsing episodic memory file '{self.episodic_memory_file}': {e}")
            return {}

    def _extract_clean_json(self, text: str) -> Dict[str, Any]:
        """Robustly extract and clean JSON from LLM response."""
        try:
            # First attempt: direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  🔧 LLM returned malformed JSON, attempting to fix... Error: {e}")
            
            # Second attempt: find JSON object and fix common issues
            try:
                # Look for JSON-like structure
                start_index = text.find('{')
                end_index = text.rfind('}')
                
                if start_index != -1 and end_index != -1:
                    json_str = text[start_index:end_index+1]
                    
                    # Fix common JSON issues
                    fixed_json = self._fix_common_json_issues(json_str)
                    
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        # Third attempt: pattern matching fallback
                        print(f"  🔧 JSON fixing failed, using pattern matching fallback...")
                        return self._extract_components_by_pattern(text)
                else:
                    # No JSON structure found, use pattern matching
                    print(f"  🔧 No JSON structure found, using pattern matching fallback...")
                    return self._extract_components_by_pattern(text)
                    
            except Exception as fix_error:
                print(f"  ❌ Failed to fix JSON formatting issues: {fix_error}")
                # Final fallback: pattern matching
                print(f"  🔧 Using pattern matching fallback...")
                return self._extract_components_by_pattern(text)
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove newlines and extra spaces
        json_str = ' '.join(json_str.split())
        
        # Fix missing quotes around property names
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix missing quotes around string values
        json_str = re.sub(r':\s*([^,\[\]]+?)(?=\s*[,}\]])', r': "\1"', json_str)
        
        # Fix array formatting
        json_str = re.sub(r'\[\s*([^\[\]]+?)\s*\]', r'["\1"]', json_str)
        
        return json_str
    
    def _extract_components_by_pattern(self, text: str) -> Dict[str, Any]:
        """Extract components using pattern matching when JSON parsing fails completely."""
        print(f"  🔧 PATTERN MATCHING FALLBACK: Extracting components manually")
        
        # Initialize default structure
        components = {
            "core_subject": "",
            "enhancements": {
                "quality_adjectives": [],
                "material_details": [],
                "light_interaction": [],
                "context": []
            }
        }
        
        # Try to extract core subject
        core_patterns = [
            r'core_subject:\s*([^,\n]+)',
            r'"core_subject":\s*"([^"]+)"',
            r'core_subject["\s]*:\s*([^,\n]+)'
        ]
        
        for pattern in core_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                core_subject = match.group(1).strip()
                # Clean up the extracted subject
                core_subject = re.sub(r'[,\n\r\t]', ' ', core_subject).strip()
                components["core_subject"] = core_subject
                print(f"    🔍 Extracted core subject: '{core_subject}'")
                break
        
        # If no core subject found, try to infer from the original prompt
        if not components["core_subject"]:
            # Look for the original prompt in the text
            prompt_match = re.search(r'Generating 3D model:\s*[\'"]([^\'"]+)[\'"]', text)
            if prompt_match:
                components["core_subject"] = prompt_match.group(1).strip()
                print(f"    🔍 Inferred core subject from prompt: '{components['core_subject']}'")
            else:
                components["core_subject"] = "unknown object"
                print(f"    ⚠️ Could not extract core subject, using default")
        
        # Try to extract quality adjectives
        adj_patterns = [
            r'quality_adjectives:\s*\[([^\]]+)\]',
            r'"quality_adjectives":\s*\[([^\]]+)\]',
            r'quality_adjectives["\s]*:\s*\[([^\]]+)\]'
        ]
        
        for pattern in adj_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                adj_text = match.group(1).strip()
                # Split by comma and clean
                adjectives = [adj.strip().strip('"\'') for adj in adj_text.split(',') if adj.strip()]
                components["enhancements"]["quality_adjectives"] = adjectives
                print(f"    🔍 Extracted quality adjectives: {adjectives}")
                break
        
        # Try to extract material details
        material_patterns = [
            r'material_details:\s*\[([^\]]+)\]',
            r'"material_details":\s*\[([^\]]+)\]',
            r'material_details["\s]*:\s*\[([^\]]+)\]'
        ]
        
        for pattern in material_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                material_text = match.group(1).strip()
                # Split by comma and clean
                materials = [mat.strip().strip('"\'') for mat in material_text.split(',') if mat.strip()]
                components["enhancements"]["material_details"] = materials
                print(f"    🔍 Extracted material details: {materials}")
                break
        
        # Try to extract light interaction
        light_patterns = [
            r'light_interaction:\s*\[([^\]]+)\]',
            r'"light_interaction":\s*\[([^\]]+)\]',
            r'light_interaction["\s]*:\s*\[([^\]]+)\]'
        ]
        
        for pattern in light_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                light_text = match.group(1).strip()
                # Split by comma and clean
                lights = [light.strip().strip('"\'') for light in light_text.split(',') if light.strip()]
                components["enhancements"]["light_interaction"] = lights
                print(f"    🔍 Extracted light interaction: {lights}")
                break
        
        # Try to extract context
        context_patterns = [
            r'context:\s*\[([^\]]+)\]',
            r'"context":\s*\[([^\]]+)\]',
            r'context["\s]*:\s*\[([^\]]+)\]'
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                context_text = match.group(1).strip()
                # Split by comma and clean
                contexts = [ctx.strip().strip('"\'') for ctx in context_text.split(',') if ctx.strip()]
                components["enhancements"]["context"] = contexts
                print(f"    🔍 Extracted context: {contexts}")
                break
        
        print(f"    🔧 Pattern matching fallback completed")
        return components

    def extract_patterns(self, prompt_to_deconstruct: str) -> Dict[str, Any]:
        """Uses the LLM to extract patterns from a prompt."""
        system_prompt = """You are a JSON generator for 3D model prompt analysis. Your ONLY task is to output valid JSON.

**CRITICAL: RESPOND ONLY WITH VALID JSON. NO TEXT, NO EXPLANATIONS, NO COMMENTARY.**

**REQUIRED FORMAT:**
{
  "core_subject": "exact subject from prompt",
  "enhancements": {
    "quality_adjectives": ["descriptive words that add quality"],
    "material_details": ["materials, textures, properties"],
    "light_interaction": ["lighting, visual effects"],
    "context": ["background, setting, additional details"]
  }
}

**EXAMPLES:**

Input: "intricate sandstone sculpture of cat lounging"
Output: {"core_subject":"sandstone sculpture of cat lounging","enhancements":{"quality_adjectives":["intricate"],"material_details":[],"light_interaction":[],"context":[]}}

Input: "stone-etched armor with leafy pattern"
Output: {"core_subject":"stone-etched armor with leafy pattern","enhancements":{"quality_adjectives":[],"material_details":[],"light_interaction":[],"context":[]}}

Input: "glowing crystal pendant with silver chain"
Output: {"core_subject":"crystal pendant with silver chain","enhancements":{"quality_adjectives":[],"material_details":["crystal","silver"],"light_interaction":["glowing"],"context":[]}}

**RULES:**
1. Core subject includes descriptive words that are part of the object name
2. Quality adjectives are separate descriptive words that add detail
3. Material details are physical properties and textures
4. Light interaction includes visual effects and lighting
5. Context includes background and setting details

**FINAL INSTRUCTION: OUTPUT ONLY VALID JSON. NOTHING ELSE.**"""

        # The _query_llm function automatically chooses between vLLM and Ollama based on configuration
        # Start with opening brace to cue JSON output
        user_prompt = f"{{ \"core_subject\": \"{prompt_to_deconstruct}\", \"enhancements\": {{"
        response_str = self.optimizer._query_llm(system_prompt, user_prompt)
        print(f"  🔍 LLM response: '{response_str}...'")
        # Use the robust JSON cleaning method
        cleaned = self._extract_clean_json(response_str)
        print(f"  🔍 Cleaned JSON: '{cleaned}'")
        return cleaned

    def find_closest_gold_prompt(self, original_prompt: str, min_similarity: float = 0.3) -> Optional[Tuple[str, str, float, float]]:
        """
        Find the closest gold prompt to the original prompt.
        Returns (gold_original, gold_prompt, gold_score, similarity) or None if no close match found.
        """
        best_match = None
        best_similarity = 0.0
        
        for gold_original, gold_data in self.gold_standard_results.items():
            # Calculate similarity between original prompts
            similarity = calculate_similarity(original_prompt, gold_original)
            
            if similarity > best_similarity and similarity >= min_similarity:
                best_similarity = similarity
                best_run = gold_data.get("method_2_hybrid_example", {})
                if best_run and "optimized_prompt" in best_run:
                    gold_prompt = best_run["optimized_prompt"]
                    gold_score = best_run["validation_results"]["validation_engine_score"]
                    best_match = (gold_original, gold_prompt, gold_score, similarity)
        
        return best_match

    def merge_components_intelligently(self, original_components: Dict[str, Any], 
                                     gold_components: Dict[str, Any], 
                                     similarity: float) -> Dict[str, Any]:
        """
        Intelligently merge components from original and gold prompts.
        ALWAYS preserve the original prompt's core subject and intent.
        Use gold prompt components only as enhancements, not replacements.
        """
        merged_components = {
            "core_subject": "",
            "enhancements": {
                "quality_adjectives": [],
                "material_details": [],
                "light_interaction": [],
                "context": []
            }
        }
        
        # CRITICAL: Always preserve the original core subject
        # The original prompt's intent should never be completely replaced
        merged_components["core_subject"] = original_components.get("core_subject", "")
        
        # DEBUG: Log what we're merging
        print(f"    🔍 MERGING DEBUG:")
        print(f"       Original core subject: '{original_components.get('core_subject', 'N/A')}'")
        print(f"       Gold core subject: '{gold_components.get('core_subject', 'N/A')}'")
        print(f"       Final core subject: '{merged_components['core_subject']}'")
        
        # Merge enhancements intelligently based on similarity and quality
        for enhancement_type in ["quality_adjectives", "material_details", "light_interaction", "context"]:
            original_items = original_components.get("enhancements", {}).get(enhancement_type, [])
            gold_items = gold_components.get("enhancements", {}).get(enhancement_type, [])
            
            merged_items = []
            
            # Always include original items (preserve intent)
            merged_items.extend(original_items)
            
            # Intelligently add gold items based on similarity and quality
            if similarity >= 0.4:  # Only use gold components if similarity is decent
                for gold_item in gold_items:
                    # Avoid duplicates
                    if gold_item.lower() not in [item.lower() for item in merged_items]:
                        # Only add if it's actually an enhancement
                        if self._is_quality_enhancement(gold_item, enhancement_type):
                            merged_items.append(gold_item)
                            print(f"         {enhancement_type}: Added '{gold_item}' from gold prompt")
                        else:
                            print(f"         {enhancement_type}: Skipped '{gold_item}' (not quality-enhancing)")
            
            merged_components["enhancements"][enhancement_type] = merged_items
        
        return merged_components
    
    def _is_quality_enhancement(self, item: str, enhancement_type: str) -> bool:
        """Determine if an enhancement item actually improves quality"""
        item_lower = item.lower()
        
        # Quality adjectives that typically improve prompts
        quality_boosters = {
            "quality_adjectives": ["intricate", "detailed", "precise", "polished", "refined", "exquisite", "meticulous"],
            "material_details": ["polished", "textured", "smooth", "rough", "metallic", "glossy", "matte"],
            "light_interaction": ["glowing", "shimmering", "reflective", "illuminated", "bright", "soft"],
            "context": ["studio", "professional", "clean", "minimal", "focused"]
        }
        
        # Check if this item is a known quality booster
        if enhancement_type in quality_boosters:
            if any(booster in item_lower for booster in quality_boosters[enhancement_type]):
                return True
        
        # Check if it's a specific, descriptive term (usually good)
        if len(item.split()) <= 2 and item_lower not in ["white", "background", "simple", "basic"]:
            return True
        
        return False

    def _items_conflict(self, item1: str, item2: str) -> bool:
        """Check if two enhancement items conflict with each other."""
        # Simple conflict detection - could be enhanced
        item1_lower = item1.lower()
        item2_lower = item2.lower()
        
        # Check for opposite adjectives
        opposites = [
            ("bright", "dark"), ("shiny", "matte"), ("smooth", "rough"),
            ("warm", "cool"), ("light", "heavy"), ("small", "large")
        ]
        
        for opp1, opp2 in opposites:
            if (opp1 in item1_lower and opp2 in item2_lower) or (opp2 in item1_lower and opp1 in item2_lower):
                return True
        
        return False

    def reconstruct_prompt(self, components: Dict[str, Any], reference_original: str = None, reference_optimized: str = None, target_original: str = None) -> str:
        """Uses the LLM to reconstruct a prompt from merged components."""
        components_json = json.dumps(components, indent=2)
        
        # DEBUG: Show exactly what we're sending to the LLM
        print(f"    🔧 RECONSTRUCTION DEBUG:")
        print(f"       Sending to LLM:")
        print(f"       Core subject: '{components.get('core_subject', 'N/A')}'")
        print(f"       Quality adjectives: {components.get('enhancements', {}).get('quality_adjectives', [])}")
        print(f"       Material details: {components.get('enhancements', {}).get('material_details', [])}")
        print(f"       Light interaction: {components.get('enhancements', {}).get('light_interaction', [])}")
        print(f"       Context: {components.get('enhancements', {}).get('context', [])}")
        
        # Create a more organic and intelligent system prompt
        system_prompt = f"""You are a JSON prompt reconstructor for 3D model generation. Your ONLY task is to output a single optimized prompt.

**CRITICAL: OUTPUT ONLY THE OPTIMIZED PROMPT. NO JSON, NO EXPLANATIONS, NO TEXT.**

**INPUT COMPONENTS:**
{components_json}

**REFERENCE EXAMPLES:**
"""
        
        # Add reference examples if provided
        if reference_original and reference_optimized:
            system_prompt += f"""Reference Original: "{reference_original}"
Reference Optimized: "{reference_optimized}"

Study how the optimized version enhances while preserving the core subject.
"""
        
        if target_original:
            system_prompt += f"""Target Original: "{target_original}"

This is what we're enhancing. Keep the same subject and intent.
"""
        
        system_prompt += f"""
**RECONSTRUCTION RULES:**
1. Start with the exact core subject: "{components.get('core_subject', '')}"
2. Add quality adjectives naturally: {components.get('enhancements', {}).get('quality_adjectives', [])}
3. Integrate material details: {components.get('enhancements', {}).get('material_details', [])}
4. Include light effects: {components.get('enhancements', {}).get('light_interaction', [])}
5. Add context details: {components.get('enhancements', {}).get('context', [])}
6. End with ", white background"

**STYLE:**
- Make it sound natural and professional
- Avoid robotic or repetitive language
- Use adjectives that actually improve the image
- Think like a 3D artist describing what they need

**EXAMPLE TRANSFORMATION:**
Input: "stone-etched armor with leafy pattern"
Enhancement: "intricately patterned stone-etched armor with leafy pattern, white background"

**FINAL OUTPUT: ONLY THE ENHANCED PROMPT, NOTHING ELSE.**"""
        
        # DEBUG: Show the full system prompt being sent to the LLM
        print(f"       🔧 Full system prompt sent to LLM:")
        # print(f"          {system_prompt[:200]}...")
        # if len(system_prompt) > 200:
        #     print(f"          ... (truncated, total length: {len(system_prompt)} chars)")
        
        # We pass an empty user prompt as the full instructions are in the system prompt
        # Add a user prompt that cues the output format
        user_prompt = f"Reconstruct prompt for: {components.get('core_subject', '')}"
        result = self.optimizer._query_llm(system_prompt, user_prompt)
        
        # DEBUG: Show what the LLM returned
        print(f"       LLM returned: '{result}'")
        
        # ENHANCED VALIDATION: Check if the LLM followed instructions
        core_subject = components.get('core_subject', '')
        if core_subject:
            # Check if core subject is preserved
            if core_subject.lower() not in result.lower():
                print(f"       ❌ LLM FAILED: Core subject '{core_subject}' not found in response")
                print(f"       🔧 Using fallback construction...")
                return self._construct_fallback_prompt(components)
            
            # Check if white background is added
            if "white background" not in result.lower():
                print(f"       ⚠️ LLM WARNING: Missing 'white background', adding it")
                result = result.rstrip() + ", white background"
                print(f"       🔧 Corrected prompt: '{result}'")
            
            # Additional validation: check if result is too short or too long
            if len(result.split()) < 3:
                print(f"       ❌ LLM FAILED: Response too short ({len(result.split())} words)")
                print(f"       🔧 Using fallback construction...")
                return self._construct_fallback_prompt(components)
            
            if len(result.split()) > 25:
                print(f"       ⚠️ LLM WARNING: Response too long ({len(result.split())} words), truncating...")
                words = result.split()
                result = ' '.join(words[:20]) + ", white background"
                print(f"       🔧 Truncated prompt: '{result}'")
        
        # Final validation: ensure the result is reasonable
        if not result or result.strip() == "":
            print(f"       ❌ LLM FAILED: Empty response")
            print(f"       🔧 Using fallback construction...")
            return self._construct_fallback_prompt(components)
        
        return result
    
    def _construct_fallback_prompt(self, components: Dict[str, Any]) -> str:
        """Construct a prompt manually when the LLM fails to follow instructions"""
        print(f"       🔧 FALLBACK: Intelligent manual prompt construction")
        
        # Get the core subject (this is critical)
        core_subject = components.get('core_subject', '')
        if not core_subject:
            print(f"       ❌ FALLBACK ERROR: No core subject found")
            return "white background"  # Minimal fallback
        
        # Start building the prompt intelligently
        prompt_parts = []
        
        # 1. Add quality adjectives at the beginning (most impactful)
        quality_adj = components.get('enhancements', {}).get('quality_adjectives', [])
        if quality_adj:
            # Filter out generic terms and add meaningful ones
            meaningful_adj = [adj for adj in quality_adj if adj.lower() not in ['simple', 'basic', 'plain']]
            if meaningful_adj:
                prompt_parts.extend(meaningful_adj)
        
        # 2. Add the core subject
        prompt_parts.append(core_subject)
        
        # 3. Add material details intelligently
        material_details = components.get('enhancements', {}).get('material_details', [])
        if material_details:
            # Filter out materials already mentioned in core subject
            core_lower = core_subject.lower()
            additional_materials = []
            for mat in material_details:
                mat_lower = mat.lower()
                # Only add if it's not already mentioned and is meaningful
                if (mat_lower not in core_lower and 
                    mat_lower not in ['white', 'background', 'simple', 'basic'] and
                    len(mat.split()) <= 2):  # Keep it simple
                    additional_materials.append(mat)
            
            if additional_materials:
                # Create a natural phrase
                if len(additional_materials) == 1:
                    prompt_parts.append(f"with {additional_materials[0]} finish")
                else:
                    prompt_parts.append(f"with {', '.join(additional_materials)} details")
        
        # 4. Add light interaction effects
        light_interaction = components.get('enhancements', {}).get('light_interaction', [])
        if light_interaction:
            meaningful_light = [light for light in light_interaction if light.lower() not in ['simple', 'basic']]
            if meaningful_light:
                if len(meaningful_light) == 1:
                    prompt_parts.append(f"with {meaningful_light[0]} lighting")
                else:
                    prompt_parts.append(f"with {', '.join(meaningful_light)} effects")
        
        # 5. Add context details
        context = components.get('enhancements', {}).get('context', [])
        if context:
            meaningful_context = [ctx for ctx in context if ctx.lower() not in ['white', 'background']]
            if meaningful_context:
                prompt_parts.extend(meaningful_context)
        
        # 6. Construct the final prompt
        if len(prompt_parts) == 1:
            # Just core subject, add a basic enhancement
            fallback_prompt = f"{core_subject}, detailed, white background"
        else:
            # Multiple parts, join naturally
            fallback_prompt = f"{' '.join(prompt_parts)}, white background"
        
        # 7. Clean up the prompt
        fallback_prompt = self._clean_fallback_prompt(fallback_prompt)
        
        # 8. Final safety check - ensure we have something valid
        if not fallback_prompt or fallback_prompt.strip() == "":
            print(f"       ❌ FALLBACK ERROR: Generated empty prompt, using minimal fallback")
            fallback_prompt = f"{core_subject}, detailed, white background"
        
        # 9. Ensure white background is present
        if "white background" not in fallback_prompt.lower():
            fallback_prompt = fallback_prompt.rstrip() + ", white background"
        
        print(f"       🔧 Fallback prompt: '{fallback_prompt}'")
        return fallback_prompt
    
    def _clean_fallback_prompt(self, prompt: str) -> str:
        """Clean up the fallback prompt to make it more natural"""
        # Remove double spaces
        prompt = ' '.join(prompt.split())
        
        # Fix common awkward phrases and duplicates
        replacements = [
            ('with with', 'with'),
            ('details details', 'details'),
            ('effects effects', 'effects'),
            ('lighting lighting', 'lighting'),
            ('finish finish', 'finish'),
            (', ,', ','),
            (' ,', ','),
            (',,', ','),
            # Fix the specific duplicate issue we saw
            ('stone-etched leafy intricately patterned stone-etched', 'intricately patterned stone-etched'),
            ('white backgroun white background', 'white background'),
        ]
        
        for old, new in replacements:
            prompt = prompt.replace(old, new)
        
        # Remove duplicate consecutive words
        words = prompt.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() != words[i-1].lower():
                cleaned_words.append(word)
        prompt = ' '.join(cleaned_words)
        
        # Ensure proper spacing around commas
        prompt = prompt.replace(' ,', ',').replace(', ', ', ')
        
        # Remove trailing comma before "white background"
        if prompt.endswith(', white background'):
            prompt = prompt[:-1] + ' white background'
        
        return prompt

    def optimize_prompt_with_reproducibility(self, original_prompt: str, min_similarity: float = 0.3, run_validation: bool = True) -> Optional[Dict[str, Any]]:
        """
        Main function to optimize a prompt using reproducibility techniques.
        Returns optimization result or None if no suitable gold prompt found.
        """
        print(f"🔍 Finding close gold prompt for: '{original_prompt[:50]}...'")
        
        # Step 1: Find the closest gold prompt
        closest_match = self.find_closest_gold_prompt(original_prompt, min_similarity)
        if not closest_match:
            print(f"  ⚠️ No close gold prompt found (similarity threshold: {min_similarity})")
            return None
        
        gold_original, gold_prompt, gold_score, similarity = closest_match
        print(f"  🏆 Found close gold prompt (similarity: {similarity:.3f}, score: {gold_score:.4f})")
        print(f"     Gold original: '{gold_original[:80]}...'")
        print(f"     Gold optimized: '{gold_prompt[:80]}...'")
        
        # Step 2: Extract components from original prompt
        print("  📝 Extracting components from original prompt...")
        original_components = self.extract_patterns(original_prompt)
        if "error" in original_components:
            print(f"  ❌ Failed to extract components from original prompt")
            return None
        
        # Step 3: Extract components from gold prompt
        print("  📝 Extracting components from gold prompt...")
        gold_components = self.extract_patterns(gold_prompt)
        if "error" in gold_components:
            print(f"  ❌ Failed to extract components from gold prompt")
            return None
        
        # Step 4: Intelligently merge components
        print(f"  🔄 Merging components (similarity: {similarity:.3f})...")
        merged_components = self.merge_components_intelligently(original_components, gold_components, similarity)
        
        # Step 5: Reconstruct the optimized prompt
        print("  🔧 Reconstructing optimized prompt...")
        print(f"     Using reference examples:")
        print(f"       Reference Original: '{gold_original[:60]}...'")
        print(f"       Reference Optimized: '{gold_prompt[:60]}...'")
        print(f"       Target Original: '{original_prompt[:60]}...'")
        
        # Pass reference examples to guide the reconstruction
        optimized_prompt = self.reconstruct_prompt(
            merged_components, 
            reference_original=gold_original,  # The original prompt from the gold standard
            reference_optimized=gold_prompt,   # The optimized version from the gold standard
            target_original=original_prompt    # The prompt we're trying to enhance
        )
        if "Error:" in optimized_prompt:
            print(f"  ❌ Failed to reconstruct prompt")
            return None
        
        print(f"  ✅ Optimization complete!")
        print(f"     Original: '{original_prompt[:80]}...'")
        print(f"     Optimized: '{optimized_prompt[:80]}...'")
        
        # Step 6: Validate the optimized prompt to get ground truth score (optional)
        if run_validation:
            print("  🎯 Running ground truth validation...")
            validation_results = self._run_validator(original_prompt, optimized_prompt)
            optimized_score = validation_results.get("validation_engine_score", 0.0)
            
            # Calculate improvement metrics
            score_improvement = optimized_score - gold_score
            improvement_percentage = ((optimized_score - gold_score) / gold_score * 100) if gold_score > 0 else 0
            
            print(f"  📊 Validation Results:")
            print(f"     Gold Score: {gold_score:.4f}")
            print(f"     Optimized Score: {optimized_score:.4f}")
            print(f"     Score Delta: {score_improvement:+.4f} ({improvement_percentage:+.1f}%)")
            
            # Determine success status
            if optimized_score >= gold_score * 0.95:
                status = "✅ SUCCESS (>= 95% of gold score)"
            elif optimized_score >= gold_score * 0.9:
                status = "⚠️ ACCEPTABLE (>= 90% of gold score)"
            else:
                status = "❌ REGRESSION (< 90% of gold score)"
            
            print(f"     Status: {status}")
        else:
            # Skip validation
            optimized_score = None
            score_improvement = None
            improvement_percentage = None
            validation_results = None
            status = "⏭️ SKIPPED (validation disabled)"
            print(f"  ⏭️ Validation skipped (use --no-validation to disable)")
        
        return {
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "gold_original": gold_original,
            "gold_prompt": gold_prompt,
            "gold_score": gold_score,
            "optimized_score": optimized_score,
            "score_improvement": score_improvement,
            "improvement_percentage": improvement_percentage,
            "similarity": similarity,
            "original_components": original_components,
            "gold_components": gold_components,
            "merged_components": merged_components,
            "validation_results": validation_results,
            "optimization_method": "reproducibility_merge",
            "status": status
        }

    def update_gold_standard_results(self, enhanced_gold_prompts: Dict[str, Any]):
        """
        Update the gold standard results with enhanced data from the orchestrator.
        This allows the reproducibility system to use optimized versions instead of just original prompts.
        
        Args:
            enhanced_gold_prompts: Dictionary of enhanced gold prompts with optimized versions
        """
        if not enhanced_gold_prompts:
            return
        
        print(f"🔄 Updating gold standard results with {len(enhanced_gold_prompts)} enhanced prompts...")
        
        # Convert enhanced format to the format expected by the reproducibility system
        for prompt, data in enhanced_gold_prompts.items():
            if 'optimized_prompt' in data and data['optimized_prompt'] != prompt:
                # This prompt has an optimized version
                optimized_prompt = data['optimized_prompt']
                score = data.get('best_score', 0.0)
                
                # Create the method_2_hybrid_example structure
                if prompt not in self.gold_standard_results:
                    self.gold_standard_results[prompt] = {}
                
                self.gold_standard_results[prompt]["method_2_hybrid_example"] = {
                    "optimized_prompt": optimized_prompt,
                    "validation_results": {
                        "validation_engine_score": score
                    }
                }
                
                # print(f"   ✅ Updated: '{prompt[:50]}...' → '{optimized_prompt[:50]}...' (score: {score:.4f})")
        
        print(f"🔄 Gold standard results updated: {len(self.gold_standard_results)} total prompts")
    
    def get_enhanced_gold_prompts(self) -> Dict[str, Any]:
        """
        Get enhanced gold prompts that include both original and optimized versions.
        This method can be called by the orchestrator to get the current state.
        """
        enhanced_prompts = {}
        
        for prompt, data in self.gold_standard_results.items():
            if 'method_2_hybrid_example' in data:
                method_data = data['method_2_hybrid_example']
                optimized_prompt = method_data.get('optimized_prompt', prompt)
                score = method_data.get('validation_results', {}).get('validation_engine_score', 0.0)
                
                enhanced_prompts[prompt] = {
                    'original_prompt': prompt,
                    'optimized_prompt': optimized_prompt,
                    'best_score': score,
                    'source': 'reproducibility_system',
                    'method': 'method_2_hybrid_example',
                    'status': 'completed',
                    'is_gold': score > 0.75
                }
        
        return enhanced_prompts

def main():
    """Test the reproducibility system with sample prompts."""
    parser = argparse.ArgumentParser(
        description="Test LLM close prompt reproducibility system.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--episodic-memory", default="episodic_logs/episodic_memory.json",
        help="Path to episodic memory file (default: episodic_logs/episodic_memory.json)"
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.3,
        help="Minimum similarity threshold for finding close prompts (default: 0.3)"
    )
    parser.add_argument(
        "--test-prompts", nargs="+",
        help="Specific prompts to test (default: uses episodic_test_prompts)"
    )
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Skip ground truth validation (faster testing)"
    )
    args = parser.parse_args()

    # Initialize the reproducibility system
    repro_system = LLMClosePromptReproducibility(args.episodic_memory)
    
    # Determine which prompts to test
    if args.test_prompts:
        test_prompts = args.test_prompts
    else:
        test_prompts = EPISODIC_TEST_PROMPTS[:5]  # Test first 5 prompts
    
    print(f"🧪 Testing reproducibility system with {len(test_prompts)} prompts...")
    print(f"   Min similarity threshold: {args.min_similarity}")
    print(f"   Gold prompts available: {len(repro_system.gold_standard_results)}")
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        
        result = repro_system.optimize_prompt_with_reproducibility(prompt, args.min_similarity, not args.no_validation)
        
        if result:
            results.append(result)
            print(f"  ✅ Success: similarity={result['similarity']:.3f}, gold_score={result['gold_score']:.4f}")
            print(f"     Gold original: '{result['gold_original'][:50]}...'")
        else:
            print(f"  ❌ Failed: no suitable gold prompt found")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"   Total prompts tested: {len(test_prompts)}")
    print(f"   Successful optimizations: {len(results)}")
    print(f"   Success rate: {len(results)/len(test_prompts)*100:.1f}%")
    
    if results:
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        avg_gold_score = sum(r['gold_score'] for r in results) / len(results)
        
        # Only calculate validation stats if validation was run
        validated_results = [r for r in results if r['optimized_score'] is not None]
        if validated_results:
            avg_optimized_score = sum(r['optimized_score'] for r in validated_results) / len(validated_results)
            avg_improvement = sum(r['score_improvement'] for r in validated_results) / len(validated_results)
            
            # Count success categories
            successes = len([r for r in validated_results if "SUCCESS" in r['status']])
            acceptables = len([r for r in validated_results if "ACCEPTABLE" in r['status']])
            regressions = len([r for r in validated_results if "REGRESSION" in r['status']])
        else:
            avg_optimized_score = None
            avg_improvement = None
            successes = acceptables = regressions = 0
        
        print(f"   Average similarity: {avg_similarity:.3f}")
        print(f"   Average gold score: {avg_gold_score:.4f}")
        
        if validated_results:
            print(f"   Average optimized score: {avg_optimized_score:.4f}")
            print(f"   Average score improvement: {avg_improvement:+.4f}")
            print(f"   Success breakdown:")
            print(f"     ✅ SUCCESS (>= 95%): {successes}/{len(validated_results)} ({successes/len(validated_results)*100:.1f}%)")
            print(f"     ⚠️ ACCEPTABLE (>= 90%): {acceptables}/{len(validated_results)} ({acceptables/len(validated_results)*100:.1f}%)")
            print(f"     ❌ REGRESSION (< 90%): {regressions}/{len(validated_results)} ({regressions/len(validated_results)*100:.1f}%)")
            
            # Show best and worst performers
            if validated_results:
                best_result = max(validated_results, key=lambda r: r['score_improvement'])
                worst_result = min(validated_results, key=lambda r: r['score_improvement'])
                
                print(f"   Best performer:")
                print(f"     '{best_result['original_prompt'][:40]}...'")
                print(f"     Gold: {best_result['gold_score']:.4f} → Optimized: {best_result['optimized_score']:.4f} (+{best_result['score_improvement']:.4f})")
                
                print(f"   Worst performer:")
                print(f"     '{worst_result['original_prompt'][:40]}...'")
                print(f"     Gold: {worst_result['gold_score']:.4f} → Optimized: {worst_result['optimized_score']:.4f} ({worst_result['score_improvement']:+.4f})")
        else:
            print(f"   ⏭️ Validation was skipped - no score comparisons available")

if __name__ == "__main__":
    main() 