#!/usr/bin/env python3
"""
Analyze Your CLIP Alignment Results
Parses the specific test results from your output and generates detailed analysis
"""

from clip_alignment_analysis_summary import CLIPAlignmentAnalyzer

def analyze_your_results():
    """Analyze the specific test results from your output"""
    
    # Your test outputs
    test_outputs = [
        # Test 1: "small yellow triangular wooden kitchen knife" vs "small yellow triangular wooden kitchen knife, highly detailed, studio lighting, isometric view, high quality materials, precise geometry, sharp details, professional photography, 8k resolution"
        """
🔍 COMPARING TWO PROMPTS ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT
======================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'small yellow triangular wooden kitchen knife, highly detailed, studio lighting, isometric view, high quality materials, precise geometry, sharp details, professional photography, 8k resolution'
Seed: 42
Total LoRA endpoints: 9 + 1 (HunyuanDiT)

📊 PROMPT COMPARISON TABLE
====================================================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'small yellow triangular wooden kitchen knife, highly detailed, studio lighting, isometric view, high quality materials, precise geometry, sharp details, professional photography, 8k resolution' (optimized version)
====================================================================================================
LoRA Endpoint        P1 Score     P2 Score     Improvement  Best     Status         
-------------------- ------------ ------------ ------------ -------- ---------------
baolei               0.7234       0.7540       +0.0307      P2       ✅ BETTER       
hunyuan              0.6595       0.6731       +0.0136      P2       ✅ BETTER       
tf2_style            0.7220       0.7345       +0.0126      P2       ✅ BETTER       
cinema               0.7764       0.7875       +0.0112      P2       ✅ BETTER       
game_assets          0.7840       0.7889       +0.0049      P2       ✅ BETTER       
live_3d              0.8112       0.7959       -0.0153      P1       ❌ WORSE        
patched_realism      0.7973       0.7771       -0.0202      P1       ❌ WORSE        
cartoon_3d           0.8147       0.7938       -0.0209      P1       ❌ WORSE        
sd15_game_icon       0.7666       0.7247       -0.0419      P1       ❌ WORSE        
default              0.8050       0.7317       -0.0732      P1       ❌ WORSE        
isometric_3d         0.8050       0.7317       -0.0732      P1       ❌ WORSE        

📈 SUMMARY STATISTICS:
   Total endpoints tested: 11
   Successful generations: 11
   Failed generations: 0
   Average P1 score: 0.7696
   Average P2 score: 0.7539
   Average improvement: -0.0156
   Best improvement: +0.0307
   Worst improvement: -0.0732

🏆 IMPROVEMENT BREAKDOWN:
   Better with P2: 5/11 (45.5%)
   Worse with P2: 6/11 (54.5%)
   Same performance: 0/11 (0.0%)

🏆 TOP 3 IMPROVEMENTS:
   1. baolei: +0.0307 (P1: 0.7234 → P2: 0.7540)
   2. hunyuan: +0.0136 (P1: 0.6595 → P2: 0.6731)
   3. tf2_style: +0.0126 (P1: 0.7220 → P2: 0.7345)
        """,
        
        # Test 2: "small yellow triangular wooden kitchen knife" vs "front view, small yellow triangular wooden kitchen knife"
        """
🔍 COMPARING TWO PROMPTS ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT
======================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'front view, small yellow triangular wooden kitchen knife'
Seed: 42
Total LoRA endpoints: 9 + 1 (HunyuanDiT)

📊 PROMPT COMPARISON TABLE
====================================================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'front view, small yellow triangular wooden kitchen knife' (optimized version)
====================================================================================================
LoRA Endpoint        P1 Score     P2 Score     Improvement  Best     Status         
-------------------- ------------ ------------ ------------ -------- ---------------
baolei               0.7234       0.8454       +0.1221      P2       ✅ BETTER       
sd15_game_icon       0.7666       0.7931       +0.0265      P2       ✅ BETTER       
hunyuan              0.6595       0.6815       +0.0220      P2       ✅ BETTER       
live_3d              0.8112       0.8308       +0.0195      P2       ✅ BETTER       
patched_realism      0.7973       0.8161       +0.0188      P2       ✅ BETTER       
tf2_style            0.7220       0.7275       +0.0056      P2       ✅ BETTER       
cartoon_3d           0.8147       0.8078       -0.0070      P1       ❌ WORSE        
game_assets          0.7840       0.7533       -0.0307      P1       ❌ WORSE        
cinema               0.7764       0.7017       -0.0746      P1       ❌ WORSE        
default              0.8050       0.6808       -0.1242      P1       ❌ WORSE        
isometric_3d         0.8050       0.6808       -0.1242      P1       ❌ WORSE        

📈 SUMMARY STATISTICS:
   Total endpoints tested: 11
   Successful generations: 11
   Failed generations: 0
   Average P1 score: 0.7696
   Average P2 score: 0.7563
   Average improvement: -0.0133
   Best improvement: +0.1221
   Worst improvement: -0.1242

🏆 IMPROVEMENT BREAKDOWN:
   Better with P2: 6/11 (54.5%)
   Worse with P2: 5/11 (45.5%)
   Same performance: 0/11 (0.0%)

🏆 TOP 3 IMPROVEMENTS:
   1. baolei: +0.1221 (P1: 0.7234 → P2: 0.8454)
   2. sd15_game_icon: +0.0265 (P1: 0.7666 → P2: 0.7931)
   3. hunyuan: +0.0220 (P1: 0.6595 → P2: 0.6815)
        """,
        
        # Test 3: "small yellow triangular wooden kitchen knife" vs "small yellow triangular wooden kitchen knife, front view"
        """
🔍 COMPARING TWO PROMPTS ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT
======================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'small yellow triangular wooden kitchen knife, front view'
Seed: 42
Total LoRA endpoints: 9 + 1 (HunyuanDiT)

📊 PROMPT COMPARISON TABLE
====================================================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'small yellow triangular wooden kitchen knife, front view' (optimized version)
====================================================================================================
LoRA Endpoint        P1 Score     P2 Score     Improvement  Best     Status         
-------------------- ------------ ------------ ------------ -------- ---------------
baolei               0.7234       0.8531       +0.1297      P2       ✅ BETTER       
sd15_game_icon       0.7666       0.7931       +0.0265      P2       ✅ BETTER       
tf2_style            0.7220       0.7408       +0.0188      P2       ✅ BETTER       
live_3d              0.8112       0.8217       +0.0105      P2       ✅ BETTER       
hunyuan              0.6595       0.6651       +0.0056      P2       ✅ BETTER       
game_assets          0.7840       0.7861       +0.0021      P2       ✅ BETTER       
patched_realism      0.7973       0.7931       -0.0042      P1       ❌ WORSE        
cartoon_3d           0.8147       0.7889       -0.0258      P1       ❌ WORSE        
cinema               0.7764       0.7331       -0.0432      P1       ❌ WORSE        
default              0.8050       0.7234       -0.0816      P1       ❌ WORSE        
isometric_3d         0.8050       0.7234       -0.0816      P1       ❌ WORSE        

📈 SUMMARY STATISTICS:
   Total endpoints tested: 11
   Successful generations: 11
   Failed generations: 0
   Average P1 score: 0.7696
   Average P2 score: 0.7656
   Average improvement: -0.0039
   Best improvement: +0.1297
   Worst improvement: -0.0816

🏆 IMPROVEMENT BREAKDOWN:
   Better with P2: 6/11 (54.5%)
   Worse with P2: 5/11 (45.5%)
   Same performance: 0/11 (0.0%)

🏆 TOP 3 IMPROVEMENTS:
   1. baolei: +0.1297 (P1: 0.7234 → P2: 0.8531)
   2. sd15_game_icon: +0.0265 (P1: 0.7666 → P2: 0.7931)
   3. tf2_style: +0.0188 (P1: 0.7220 → P2: 0.7408)
        """,
        
        # Test 4: "small yellow triangular wooden kitchen knife" vs "small yellow triangular wooden kitchen knife, 3D game asset, isometric view"
        """
🔍 COMPARING TWO PROMPTS ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT
======================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'small yellow triangular wooden kitchen knife, 3D game asset, isometric view'
Seed: 42
Total LoRA endpoints: 9 + 1 (HunyuanDiT)

📊 PROMPT COMPARISON TABLE
====================================================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: 'small yellow triangular wooden kitchen knife, 3D game asset, isometric view' (optimized version)
====================================================================================================
LoRA Endpoint        P1 Score     P2 Score     Improvement  Best     Status         
-------------------- ------------ ------------ ------------ -------- ---------------
hunyuan              0.6595       0.7254       +0.0659      P2       ✅ BETTER       
baolei               0.7234       0.7743       +0.0509      P2       ✅ BETTER       
cinema               0.7764       0.8238       +0.0474      P2       ✅ BETTER       
tf2_style            0.7220       0.7533       +0.0314      P2       ✅ BETTER       
game_assets          0.7840       0.7980       +0.0140      P2       ✅ BETTER       
live_3d              0.8112       0.8147       +0.0035      P2       ✅ BETTER       
patched_realism      0.7973       0.7987       +0.0014      P2       ✅ BETTER       
cartoon_3d           0.8147       0.8092       -0.0056      P1       ❌ WORSE        
default              0.8050       0.7429       -0.0621      P1       ❌ WORSE        
isometric_3d         0.8050       0.7429       -0.0621      P1       ❌ WORSE        
sd15_game_icon       0.7666       0.6923       -0.0743      P1       ❌ WORSE        

📈 SUMMARY STATISTICS:
   Total endpoints tested: 11
   Successful generations: 11
   Failed generations: 0
   Average P1 score: 0.7696
   Average P2 score: 0.7705
   Average improvement: +0.0010
   Best improvement: +0.0659
   Worst improvement: -0.0743

🏆 IMPROVEMENT BREAKDOWN:
   Better with P2: 7/11 (63.6%)
   Worse with P2: 4/11 (36.4%)
   Same performance: 0/11 (0.0%)

🏆 TOP 3 IMPROVEMENTS:
   1. hunyuan: +0.0659 (P1: 0.6595 → P2: 0.7254)
   2. baolei: +0.0509 (P1: 0.7234 → P2: 0.7743)
   3. cinema: +0.0474 (P1: 0.7764 → P2: 0.8238)
        """,
        
        # Test 5: "small yellow triangular wooden kitchen knife" vs " 3D game asset, isometric viewsmall yellow triangular wooden kitchen knife, photoshoot, 8k resolution"
        """
🔍 COMPARING TWO PROMPTS ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT
======================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: ' 3D game asset, isometric viewsmall yellow triangular wooden kitchen knife, photoshoot, 8k resolution'
Seed: 42
Total LoRA endpoints: 9 + 1 (HunyuanDiT)

📊 PROMPT COMPARISON TABLE
====================================================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: ' 3D game asset, isometric viewsmall yellow triangular wooden kitchen knife, photoshoot, 8k resolution' (optimized version)
====================================================================================================
LoRA Endpoint        P1 Score     P2 Score     Improvement  Best     Status         
-------------------- ------------ ------------ ------------ -------- ---------------
baolei               0.7234       0.8252       +0.1018      P2       ✅ BETTER       
cinema               0.7764       0.7973       +0.0209      P2       ✅ BETTER       
game_assets          0.7840       0.7799       -0.0042      P1       ❌ WORSE        
cartoon_3d           0.8147       0.8071       -0.0077      P1       ❌ WORSE        
tf2_style            0.7220       0.7038       -0.0181      P1       ❌ WORSE        
patched_realism      0.7973       0.7764       -0.0209      P1       ❌ WORSE        
live_3d              0.8112       0.7771       -0.0342      P1       ❌ WORSE        
sd15_game_icon       0.7666       0.6857       -0.0809      P1       ❌ WORSE        
default              0.8050       0.7042       -0.1008      P1       ❌ WORSE        
isometric_3d         0.8050       0.7042       -0.1008      P1       ❌ WORSE        
hunyuan              0.6595       0.2822       -0.3774      P1       ❌ WORSE        

📈 SUMMARY STATISTICS:
   Total endpoints tested: 11
   Successful generations: 11
   Failed generations: 0
   Average P1 score: 0.7696
   Average P2 score: 0.7130
   Average improvement: -0.0566
   Best improvement: +0.1018
   Worst improvement: -0.3774

🏆 IMPROVEMENT BREAKDOWN:
   Better with P2: 2/11 (18.2%)
   Worse with P2: 9/11 (81.8%)
   Same performance: 0/11 (0.0%)

🏆 TOP 3 IMPROVEMENTS:
   1. baolei: +0.1018 (P1: 0.7234 → P2: 0.8252)
   2. cinema: +0.0209 (P1: 0.7764 → P2: 0.7973)
   3. game_assets: -0.0042 (P1: 0.7840 → P2: 0.7799)
        """,
        
        # Test 6: "small yellow triangular wooden kitchen knife" vs " isometric view, small yellow triangular wooden kitchen knife, photoshoot, 8k resolution"
        """
🔍 COMPARING TWO PROMPTS ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT
======================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: ' isometric view, small yellow triangular wooden kitchen knife, photoshoot, 8k resolution'
Seed: 42
Total LoRA endpoints: 9 + 1 (HunyuanDiT)

📊 PROMPT COMPARISON TABLE
====================================================================================================
Prompt 1: 'small yellow triangular wooden kitchen knife'
Prompt 2: ' isometric view, small yellow triangular wooden kitchen knife, photoshoot, 8k resolution' (optimized version)
====================================================================================================
LoRA Endpoint        P1 Score     P2 Score     Improvement  Best     Status         
-------------------- ------------ ------------ ------------ -------- ---------------
baolei               0.7234       0.7980       +0.0746      P2       ✅ BETTER       
sd15_game_icon       0.7666       0.8085       +0.0419      P2       ✅ BETTER       
patched_realism      0.7973       0.8280       +0.0307      P2       ✅ BETTER       
game_assets          0.7840       0.7973       +0.0133      P2       ✅ BETTER       
cartoon_3d           0.8147       0.8245       +0.0098      P2       ✅ BETTER       
live_3d              0.8112       0.8168       +0.0056      P2       ✅ BETTER       
tf2_style            0.7220       0.7139       -0.0080      P1       ❌ WORSE        
cinema               0.7764       0.7624       -0.0140      P1       ❌ WORSE        
default              0.8050       0.7687       -0.0363      P1       ❌ WORSE        
isometric_3d         0.8050       0.7687       -0.0363      P1       ❌ WORSE        
hunyuan              0.6595       0.4987       -0.1608      P1       ❌ WORSE        

📈 SUMMARY STATISTICS:
   Total endpoints tested: 11
   Successful generations: 11
   Failed generations: 0
   Average P1 score: 0.7696
   Average P2 score: 0.7623
   Average improvement: -0.0072
   Best improvement: +0.0746
   Worst improvement: -0.1608

🏆 IMPROVEMENT BREAKDOWN:
   Better with P2: 6/11 (54.5%)
   Worse with P2: 5/11 (45.5%)
   Same performance: 0/11 (0.0%)

🏆 TOP 3 IMPROVEMENTS:
   1. baolei: +0.0746 (P1: 0.7234 → P2: 0.7980)
   2. sd15_game_icon: +0.0419 (P1: 0.7666 → P2: 0.8085)
   3. patched_realism: +0.0307 (P1: 0.7973 → P2: 0.8280)
        """
    ]
    
    # Create analyzer and parse all test outputs
    analyzer = CLIPAlignmentAnalyzer()
    
    print("🔍 PARSING YOUR CLIP ALIGNMENT TEST RESULTS...")
    print("=" * 80)
    
    for i, output in enumerate(test_outputs, 1):
        try:
            session = analyzer.parse_test_output(output)
            print(f"✅ Parsed Test {i}: {session.session_id}")
        except Exception as e:
            print(f"❌ Failed to parse Test {i}: {e}")
    
    print(f"\n📊 PARSED {len(analyzer.test_sessions)} TEST SESSIONS")
    print(f"📊 TOTAL RESULTS: {len(analyzer.all_results)}")
    print(f"🎯 HIGH SCORES (>0.3): {len(analyzer.high_score_results)}")
    
    # Generate comprehensive analysis
    print("\n" + "="*100)
    print("🎯 COMPREHENSIVE ANALYSIS OF YOUR CLIP ALIGNMENT RESULTS")
    print("="*100)
    
    # High score analysis
    print("\n" + analyzer.generate_high_score_analysis())
    
    # LoRA performance analysis
    print("\n" + analyzer.generate_lora_performance_analysis())
    
    # Prompt effectiveness analysis
    print("\n" + analyzer.generate_prompt_effectiveness_analysis())
    
    # Comprehensive summary
    print("\n" + analyzer.generate_comprehensive_summary())
    
    # Additional insights
    print("\n" + "="*100)
    print("💡 KEY INSIGHTS FROM YOUR RESULTS")
    print("="*100)
    
    # Find the highest scores
    if analyzer.high_score_results:
        highest_p1 = max(analyzer.high_score_results, key=lambda x: x.p1_score)
        highest_p2 = max(analyzer.high_score_results, key=lambda x: x.p2_score)
        best_improvement = max(analyzer.high_score_results, key=lambda x: x.improvement)
        
        print(f"\n🏆 HIGHEST SCORES ACHIEVED:")
        print(f"   Best P1 Score: {highest_p1.p1_score:.4f} ({highest_p1.lora_endpoint})")
        print(f"   Best P2 Score: {highest_p2.p2_score:.4f} ({highest_p2.lora_endpoint})")
        print(f"   Best Improvement: {best_improvement.improvement:+.4f} ({best_improvement.lora_endpoint})")
        
        print(f"\n📝 PROMPT STRATEGIES THAT WORKED:")
        print(f"   Base Prompt: '{highest_p2.prompt1}'")
        print(f"   Optimized Prompt: '{highest_p2.prompt2}'")
        print(f"   Best LoRA: {highest_p2.lora_endpoint}")
        
        # Analyze which LoRAs perform best for high scores
        lora_high_scores = {}
        for result in analyzer.high_score_results:
            if result.lora_endpoint not in lora_high_scores:
                lora_high_scores[result.lora_endpoint] = []
            lora_high_scores[result.lora_endpoint].append(result.p2_score)
        
        print(f"\n🎨 LoRA PERFORMANCE FOR HIGH SCORES (>0.3):")
        for lora, scores in sorted(lora_high_scores.items(), key=lambda x: max(x[1]), reverse=True):
            print(f"   {lora}: {len(scores)} high scores, best: {max(scores):.4f}, avg: {sum(scores)/len(scores):.4f}")

if __name__ == "__main__":
    analyze_your_results() 