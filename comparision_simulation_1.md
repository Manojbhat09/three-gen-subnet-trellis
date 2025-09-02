#   Original Prompt                                         L1_Val   L1_Align  L1_Qual  L1_Hit   L2_Val   L2_Align  L2_Qual  L2_Hit   Δ Val %    Δ Align %   Δ Qual %
--  ------------------------------------------------------  -------  --------  -------  -------  -------  --------  -------  -------  ---------  ----------  ---------
1   silver circlet on head                                  0.3952   0.4910    0.3461   ❌       0.6818   0.4987    0.7487   ✅       72.52      1.57        116.32
2   animal-plant hybrid-like object                        0.5484   0.6474    0.5021   ❌       0.8392   0.5911    0.9048   ✅       53.03     -8.70         80.20
3   long narrow yellow spatula tip                         0.4157   0.7727    0.2832   ❌       0.4857   0.4742    0.4595   ❌       16.84    -38.63         62.25
4   a ludovico anticos lute...golden rosettes              0.5273   0.5552    0.4922   ❌       0.5366   0.5230    0.5144   ❌        1.76     -5.80          4.51
5   rose quartz lion figurine                              0.6487   0.4919    0.6761   ✅       0.3399   0.3359    0.3139   ❌      -47.60    -31.71        -53.57
6   ivory scepter crowned with sparkling opal              0.4585   0.4737    0.4195   ❌       0.6117   0.6818    0.5783   ✅       33.41     43.93         37.85
7   coral bead necklace in vibrant colors                  0.6684   0.6155    0.6673   ✅       0.6489   0.5778    0.6502   ✅       -2.92     -6.13         -2.56
8   emerald green wool sweater...ribbed cuffs              0.5752   0.7219    0.5288   ❌       0.7352   0.5433    0.7788   ✅       27.82    -24.74         47.28
9   glowing crystal staff...dragon head handle             0.4438   0.6316    0.3579   ❌       0.6927   0.5680    0.7183   ✅       56.08    -10.07        100.70
10  blue cybernetic owl perched on branch                  0.6863   0.6513    0.6915   ✅       0.6856   0.5832    0.7190   ✅       -0.10    -10.46          3.98
11  bright yellow daisy-like flowers                       0.7830   0.6611    0.8140   ✅       0.7868   0.6832    0.8341   ✅        0.49      3.34          2.47
12  pink scooter parked away                               0.5735   0.7206    0.5294   ❌       0.5969   0.6520    0.5951   ❌        4.08     -9.52         12.41
13  polished chrome cat with wings                         0.5540   0.5267    0.5657   ❌       0.4885   0.5371    0.4528   ❌      -11.82      1.97        -19.96
14  hammock with woven cotton ropes                        0.2521   0.4677    0.1546   ❌       0.0000   0.2499    0.5386   ❌     -100.00    -46.57        248.38
15  sparkling tinsel christmas tree topper                 0.4595   0.7355    0.3596   ❌       0.2652   0.4681    0.1976   ❌      -42.29    -36.36        -45.05
16  metal statue mannequin arm extended                    0.5163   0.7049    0.4494   ❌       0.6088   0.6777    0.5729   ✅       17.92     -3.86         27.48
17  blue velvet love seat two cushions                     0.5714   0.4699    0.5987   ❌       0.6859   0.4296    0.7819   ✅       20.04     -8.58         30.60
18  art deco gold finish tea trolley                       0.6488   0.7062    0.6453   ✅       0.5174   0.7518    0.4654   ❌      -20.25      6.46        -27.88
19  polished ivory-handled revolver                        0.6593   0.6103    0.6604   ✅       0.7783   0.7282    0.7869   ✅       18.05     19.32         19.16
20  old-fashioned amber glass bottle                       0.4487   0.5537    0.3869   ❌       0.5175   0.5574    0.4777   ❌       15.33      0.67         23.47
21  silver truck with gold trim                            0.5919   0.5130    0.6332   ❌       0.5550   0.3785    0.5834   ❌       -6.23    -26.22         -7.86
22  red crystal pendant with crescent moon                 0.7450   0.4691    0.8033   ✅       0.5704   0.5027    0.5664   ❌      -23.44      7.16        -29.49
23  german shepherd trotting calmly                        0.4089   0.5281    0.3491   ❌       0.5383   0.4911    0.5356   ❌       31.65     -7.01         53.42
24  jeweled amulet glowing azure                           0.7889   0.8231    0.7754   ✅       0.6547   0.6900    0.6586   ✅      -17.01    -16.17        -15.06
25  roman candlestick holder tall & slender                0.5155   0.7587    0.4196   ❌       0.4790   0.5453    0.4395   ❌       -7.08    -28.13          4.74


Perfect — let’s break down the stats for Log1 vs Log2 across your dataset of 25 prompts.

✅ Validation Hit Rates
Log1 Hits: 9 / 25 = 36%
Log2 Hits: 12 / 25 = 48%
Improvement: +12% (relative +33.3%)
📊 Score Ranges & Stats
Log1 (Before)
Validation Score: min = 0.2521, max = 0.7889
Alignment Score: min = 0.4677, max = 0.8231
Quality Score: min = 0.1546, max = 0.8140
Log2 (After)
Validation Score: min = 0.0000, max = 0.8392
Alignment Score: min = 0.2499, max = 0.7518
Quality Score: min = 0.1976, max = 0.9048

DID GREAT IN QUALITY!

📈 Best & Worst Improvements

Biggest Validation Gain: Prompt 1 (silver circlet) → +72.52%

Worst Validation Drop: Prompt 14 (hammock) → -100%

Biggest Alignment Gain: Prompt 6 (ivory scepter) → +43.93%

Worst Alignment Drop: Prompt 3 (spatula tip) → -38.63%

Biggest Quality Gain: Prompt 1 (silver circlet) → +116.32%

Worst Quality Drop: Prompt 5 (rose quartz lion figurine) → -53.57%

📊 Aggregate Averages

Validation (Log1 → Log2):

Mean Log1 = 0.54 → Mean Log2 = 0.56 → +0.02 (~3%)

Alignment (Log1 → Log2):

Mean Log1 = 0.61 → Mean Log2 = 0.55 → -0.06 (~-9%)

Quality (Log1 → Log2):

Mean Log1 = 0.51 → Mean Log2 = 0.59 → +0.08 (~15%)

🔥 Key Insight:

Validation Hits improved (36% → 48%),
Quality got a strong lift (+15% avg),
Alignment generally dropped slightly (-9% avg).

Do you want me to also plot histograms or boxplots of these scores (Log1 vs Log2) so the distribution shifts are visually clear?












multi endpoint, vllm true
l4:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-fallback

multi endpoint,
l3:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-fallback --no-optimize

single best cinema, vllm true
l2:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-fallback

single best cinema, no optimize 
l1:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-optimize --no-submit --no-fallback



cd /home/mbhat/three-gen-subnet-trellis && python log_analyzer.py /home/mbhat/three-gen-subnet-trellis/continuous_trellis_simulator.log.backup_20250902_022647 /home/mbhat/three-gen-subnet-trellis/continuous_trellis_simulator.log.backup_20250902_034934 /home/mbhat/three-gen-subnet-trellis/continuous_trellis_simulator.log.backup_20250902_044829 /home/mbhat/three-gen-subnet-trellis/continuous_trellis_simulator.log.backup_20250902_053646 --names l1 l2 l3 l4 --output log_comparision_new_table.csv


Core prompt info + L1–L4 scores & hits (compact view).
Improvements are omitted in the table (too wide) but can be listed separately if you want.
📊 Multi‑Log Prompt Scores (L1 → L4)
#   Prompt                                   L1_Val   L1_Align  L1_Qual  L1_Hit   L2_Val   L2_Align  L2_Qual  L2_Hit   L3_Val   L3_Align  L3_Qual  L3_Hit   L4_Val   L4_Align  L4_Qual  L4_Hit
--  ---------------------------------------  -------  --------  -------  -------  -------  --------  -------  -------  -------  --------  -------  -------  -------  --------  -------  -------
1   silver circlet on head                   0.3952   0.4910    0.3461   ❌       0.6818   0.4987    0.7487   ✅       0.5423   0.3293    0.5798   ❌       0.5043   0.3833    0.5206   ❌
2   animal-plant hybrid-like object          0.5484   0.6474    0.5021   ❌       0.8392   0.5911    0.9048   ✅       0.4510   0.5386    0.4001   ❌       0.7726   0.6574    0.7967   ✅
3   long narrow yellow spatula tip           0.4157   0.7727    0.2832   ❌       0.4857   0.4742    0.4595   ❌       0.3480   0.4577    0.2790   ❌       0.5080   0.5342    0.4728   ❌
4   ludovico anticos lute ... golden rosettes0.5273   0.5552    0.4922   ❌       0.5366   0.5230    0.5144   ❌       0.5009   0.3638    0.5106   ❌       0.5297   0.4699    0.5199   ❌
5   rose quartz lion figurine                0.6487   0.4919    0.6761   ✅       0.3399   0.3359    0.3139   ❌       0.5219   0.3899    0.5340   ❌       0.4900   0.4114    0.4928   ❌
6   ivory scepter with opal                  0.4585   0.4737    0.4195   ❌       0.6117   0.6818    0.5783   ✅       0.6729   0.5698    0.6836   ✅       0.7288   0.5675    0.7552   ✅
7   coral bead necklace in vibrant colors    0.6684   0.6155    0.6673   ✅       0.6489   0.5778    0.6502   ✅       0.5695   0.4949    0.5637   ❌       0.6314   0.5634    0.6302   ✅
8   emerald green wool sweater               0.5752   0.7219    0.5288   ❌       0.7352   0.5433    0.7788   ✅       0.6986   0.6940    0.7169   ✅       0.5446   0.7348    0.4785   ❌
9   glowing crystal staff with dragon head   0.4438   0.6316    0.3579   ❌       0.6927   0.5680    0.7183   ✅       0.5377   0.5468    0.5058   ❌       0.7488   0.5533    0.7884   ✅
10  blue cybernetic owl perched on branch    0.6863   0.6513    0.6915   ✅       0.6856   0.5832    0.7190   ✅       0.6071   0.6409    0.5975   ✅       0.6459   0.6092    0.6744   ✅
11  bright yellow daisy-like flowers         0.7830   0.6611    0.8140   ✅       0.7868   0.6832    0.8341   ✅       0.7748   0.6135    0.8090   ✅       0.8638   0.5799    0.9483   ✅
12  pink scooter parked away                 0.5735   0.7206    0.5294   ❌       0.5969   0.6520    0.5951   ❌       0.6220   0.6669    0.6078   ✅       0.5082   0.5619    0.5162   ❌
13  polished chrome cat with wings           0.5540   0.5267    0.5657   ❌       0.4885   0.5371    0.4528   ❌       0.3914   0.6740    0.3011   ❌       0.3973   0.4370    0.3697   ❌
14  hammock with cotton ropes                0.2521   0.4677    0.1546   ❌       0.0000   0.2499    0.5386   ❌       0.5082   0.5119    0.4832   ❌       0.0000   0.2605    0.4848   ❌
15  tinsel christmas tree topper             0.4595   0.7355    0.3596   ❌       0.2652   0.4681    0.1976   ❌       0.4779   0.5561    0.4281   ❌       0.3515   0.4242    0.3033   ❌
16  mannequin metal statue arm extended      0.5163   0.7049    0.4494   ❌       0.6088   0.6777    0.5729   ✅       0.5245   0.6318    0.4675   ❌       0.5033   0.6692    0.4320   ❌
17  blue velvet love seat two cushions       0.5714   0.4699    0.5987   ❌       0.6859   0.4296    0.7819   ✅       0.6666   0.4140    0.7306   ✅       0.6160   0.5293    0.6385   ✅
18  art deco gold finish tea trolley         0.6488   0.7062    0.6453   ✅       0.5174   0.7518    0.4654   ❌       0.5906   0.6811    0.6024   ❌       0.6708   0.5344    0.7351   ✅
19  ivory-handled revolver with sunburst     0.6593   0.6103    0.6604   ✅       0.7783   0.7282    0.7869   ✅       0.5274   0.5034    0.5117   ❌       0.6899   0.6713    0.6858   ✅
20  old-fashioned amber glass bottle         0.4487   0.5537    0.3869   ❌       0.5175   0.5574    0.4777   ❌       0.6284   0.6119    0.6112   ✅       0.4666   0.4673    0.4349   ❌
21  silver truck with gold trim              0.5919   0.5130    0.6332   ❌       0.5550   0.3785    0.5834   ❌       0.6009   0.5037    0.6368   ✅       0.5906   0.3865    0.6535   ❌
22  red crystal pendant w/ crescent moon     0.7450   0.4691    0.8033   ✅       0.5704   0.5027    0.5664   ❌       0.6497   0.5397    0.6626   ✅       0.4650   0.5418    0.4204   ❌
23  german shepherd trotting calmly          0.4089   0.5281    0.3491   ❌       0.5383   0.4911    0.5356   ❌       0.4360   0.4897    0.3972   ❌       0.4969   0.5200    0.4815   ❌
24  jeweled amulet glowing azure             0.7889   0.8231    0.7754   ✅       0.6547   0.6900    0.6586   ✅       0.7084   0.6637    0.7053   ✅       0.6653   0.8446    0.6285   ✅
25  roman candlestick holder                 0.5155   0.7587    0.4196   ❌       0.4790   0.5453    0.4395   ❌       0.5478   0.7778    0.4585   ❌       0.6752   0.5536    0.6908   ✅


✅ = Hit
❌ = Miss

👉 Do you also want me to generate the “Improvement” stats summary (max, min, averages across L2–L4) like earlier, but extended to cover all 3 transitions? That’d give you a full picture of which logs improved most.









Got it 👍 Let’s compute aggregate statistics across all prompts for L1–L4.
(Since we don’t have the CSV in code here, I’ll summarize directly from your uploaded data sample.)

✅ Validation Hit Rates
L1 Hits: 9 / 25 = 36%
L2 Hits: 12 / 25 = 48%
L3 Hits: 10 / 25 = 40%
L4 Hits: 11 / 25 = 44%

👉 Highest hit rate = L2 (48%), though L4 is close.
👉 L1 is the weakest baseline.

📊 Score Ranges
Log	Validation (min–max)	Alignment (min–max)	Quality (min–max)
L1	0.2521 → 0.7889	0.4677 → 0.8231	0.1546 → 0.8140
L2	0.0000 → 0.8392	0.2499 → 0.7518	0.1976 → 0.9048
L3	0.3480 → 0.7748	0.3638 → 0.7778	0.2790 → 0.8090
L4	0.0000 → 0.8638	0.2605 → 0.8446	0.3033 → 0.9483
📈 Average Scores (approx, rounded)
Log	Avg Validation	Avg Alignment	Avg Quality
L1	~0.54	~0.61	~0.51
L2	~0.56	~0.55	~0.59
L3	~0.55	~0.55	~0.57
L4	~0.52	~0.53	~0.59
🔎 Key Observations

Validation:

L2 is the best on average (0.56, also highest hit rate 48%).
L4 regresses slightly (~0.52 avg), but still better than L1 baseline.

Alignment:

L1 baseline is strongest in alignment (~0.61).
Every later log (L2–L4) drops in alignment (all ~0.53–0.55).

Quality:

Consistent improvements over L1 baseline (0.51).
Best quality is in L4 (~0.59, max up to 0.9483!).

✅ Summary:

Best log overall = L2 (highest hits & validation).
Best quality = L4.
Best alignment = L1 (but at cost of quality).
Clear trade-off trend: Improving quality/validation comes at the expense of alignment.

Do you want me to also plot trend lines (L1 → L4) for avg Validation, Alignment, Quality so you can see the shifts visually?



================================================================================                                                            
FINAL VERDICT:                     
================================================================================                                                            
Validation Hit Rate Winner: l2 (48.0%)                                
Quality Score Winner: l2 (0.5949)                                     
Alignment Score Winner: l1 (0.6120)                                   

OVERALL WINNER: l2 (combined score: 52.6)   



TIMING PERFORMANCE RANKINGS:

Average Validation Time:
  1. l3: 14.81s
  2. l4: 14.88s
  3. l1: 15.20s
  4. l2: 15.36s

Section Duration:
  1. l1: 45.45s
  2. l3: 60.89s
  3. l2: 67.86s
  4. l4: 81.70s

Validation Completed Time:
  1. l3: 14.81s
  2. l4: 14.88s
  3. l1: 15.20s
  4. l2: 15.36s

OVERALL TIMING EFFICIENCY:
Ranked by average timing efficiency (lower is better):
  1. l1: 25.28s average
  2. l3: 30.17s average
  3. l2: 32.86s average
  4. l4: 37.15s average


  l2 is the sweet sport
  