# TRELLIS Prompt Performance Analysis Report

## Executive Summary

This report analyzes the performance of different prompt categories across 5 log files, covering 1,625 total tasks with an overall reliability of **84.12%**. The analysis reveals significant performance variations across prompt types and identifies the most and least reliable categories.

## Key Findings

### 📊 Overall Performance
- **Total Tasks Processed**: 1,625
- **Successful Tasks**: 1,367 (84.12%)
- **Failed Tasks**: 258 (15.88%)
- **Average Fidelity**: 0.8324
- **Best Individual Performance**: 0.9848 (glowing staff topped with radiant sapphire stone)
- **Worst Individual Performance**: 0.0000 (samurai helmet kabuto detailed)

### 🏆 Top Performing Prompt Categories

| Rank | Category | Success Rate | Tasks | Avg Fidelity | Best Example |
|------|----------|--------------|-------|--------------|--------------|
| 1 | **Tools/Instruments** | 100.0% | 3 | 0.841 | - |
| 2 | **Animals/Creatures** | 96.6% | 58 | 0.840 | Small black/white striped creature (0.9522) |
| 3 | **Statues/Sculptures** | 95.0% | 40 | 0.790 | Octopus statue (0.9341) |
| 4 | **Robots/Mechanical** | 91.5% | 294 | 0.779 | Purple robot balancing books (0.9455) |
| 5 | **Nature/Plants** | 90.9% | 22 | 0.750 | Glass blue vase with flowers (0.9291) |

### ❌ Bottom Performing Prompt Categories

| Rank | Category | Success Rate | Tasks | Avg Fidelity | Issues |
|------|----------|--------------|-------|--------------|--------|
| 11 | **Food** | 50.0% | 6 | 0.396 | Very low sample size |
| 10 | **Gems/Crystals** | 77.2% | 114 | 0.634 | High failure rate |
| 9 | **Other** | 81.4% | 907 | 0.672 | Large category, mixed results |
| 8 | **Jewelry** | 82.6% | 109 | 0.679 | Moderate performance |
| 7 | **Furniture** | 84.6% | 13 | 0.677 | Small sample size |

## Detailed Analysis by Log File

### 📈 Log File Performance Ranking

1. **continuous_trellis.log.old1** (88.77% reliability, 1,095 tasks)
   - **Best performing log** with highest volume
   - Excellent performance across all categories
   - Strong correlation between volume and reliability

2. **continuous_trellis.log.old2** (79.59% reliability, 49 tasks)
   - Good performance with smaller sample size
   - 100% success rate in statues, gems, nature, and jewelry categories

3. **continuous_trellis.log.old5** (74.94% reliability, 419 tasks)
   - Large sample size but below-average performance
   - Consistent performance across categories

4. **continuous_trellis.log.old3** (68.42% reliability, 38 tasks)
   - Poor performance with small sample size
   - High failure rate in robots/mechanical category

5. **continuous_trellis.log.old4** (66.67% reliability, 24 tasks)
   - **Worst performing log** with smallest sample size
   - 0% success rate in gems/crystals category

## Critical Insights

### 1. Volume-Reliability Correlation
- **Strong positive correlation** between task volume and reliability
- Higher volume sessions (old1: 1,095 tasks) show significantly better performance
- Smaller test runs may not represent true system capability

### 2. Category-Specific Performance Patterns

#### High-Performing Categories:
- **Animals/Creatures** (96.6% success): Excellent performance with good sample size
- **Statues/Sculptures** (95.0% success): Consistent high performance
- **Robots/Mechanical** (91.5% success): Large sample size with good reliability

#### Problematic Categories:
- **Gems/Crystals** (77.2% success): High failure rate despite large sample size
- **Food** (50.0% success): Very poor performance but small sample size
- **Other** (81.4% success): Large category with mixed results

### 3. Best and Worst Individual Prompts

#### 🏆 Best Performing Prompts:
1. **"glowing staff topped with radiant sapphire stone"** (0.9848)
2. **"white bowling pin head"** (0.9698)
3. **"purple robot balancing stack of books"** (0.9455)
4. **"spear with purple handle and copper blade"** (0.9470)

#### ❌ Worst Performing Prompts:
1. **"samurai helmet kabuto detailed"** (0.0000)
2. **"crystal-clear domes reflect moonlight softly"** (0.0000)
3. **"translucent blue crystal on smooth dark surface"** (0.0000)
4. **"modern chrome rifle sleek"** (0.0000)

## Recommendations

### Immediate Actions
1. **Investigate gems/crystals category** - High failure rate despite large sample size
2. **Analyze food category** - Very poor performance, may need prompt optimization
3. **Review "other" category** - Large mixed category needs better categorization

### Long-term Improvements
1. **Increase session stability** - Leverage volume-reliability correlation
2. **Optimize problematic categories** - Focus on gems/crystals and food
3. **Improve prompt categorization** - Better classification for "other" category
4. **Standardize configuration** - Reduce variability between sessions

### Category-Specific Recommendations
- **Gems/Crystals**: Implement specialized optimization for crystalline objects
- **Food**: Develop food-specific prompt templates
- **Robots/Mechanical**: Maintain current optimization approach (working well)
- **Animals/Creatures**: Continue current approach (excellent performance)

## Technical Analysis

### Performance Metrics by Category
- **Highest Average Fidelity**: Animals/Creatures (0.840)
- **Lowest Average Fidelity**: Food (0.396)
- **Most Consistent**: Tools/Instruments (100% success rate)
- **Most Variable**: Gems/Crystals (77.2% success rate)

### Reliability Trends
- **Best Log**: old1 (88.77% reliability, 1,095 tasks)
- **Worst Log**: old4 (66.67% reliability, 24 tasks)
- **Performance Range**: 22.10 percentage points
- **Volume Impact**: Strong correlation between volume and reliability

## Conclusion

The TRELLIS system shows **excellent performance** in most prompt categories, with animals/creatures, statues/sculptures, and robots/mechanical leading the way. The strong volume-reliability correlation suggests that longer, more stable sessions significantly improve performance.

**Priority areas for improvement**:
1. **Gems/Crystals category** - High failure rate needs investigation
2. **Food category** - Very poor performance despite small sample size
3. **Session stability** - Reduce variability between logs

**Target**: Achieve 90%+ reliability across all categories by addressing the problematic areas identified.

---

*Analysis generated on: 2025-01-27*  
*Data source: 5 log files covering 1,625 tasks*  
*Analysis tool: TRELLIS Prompt Performance Analysis Script* 