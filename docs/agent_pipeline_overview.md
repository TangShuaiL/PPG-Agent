# PPG Agent Pipeline 完整梳理

> 梳理时间：2026-08-05
> 覆盖范围：`ppg_agent/` 全部模块

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_agent() 入口                              │
│  输入: dataset, subject_id, seg_idx, query                           │
│  输出: { output, context, messages, full_tool_outputs }              │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  create_ppg_agent() → create_agent(model, tools, system_prompt,     │
│                                   middleware)                        │
│  LLM: Qwen2.5-14B-Instruct (via vLLM + ChatOpenAI)                  │
│  Middleware: QwenToolCallMiddleware (修复 <tool_call> XML 格式)      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent 决策循环 (LangChain)                         │
│                                                                      │
│  LLM 思考 → 决定调用哪个工具 → Middleware 拦截修复格式                 │
│         → 工具执行 → 结果返回 LLM → LLM 继续思考/输出最终答案          │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │工具层     │  │数据层     │  │知识层     │
        │14个工具   │  │PPGData  │  │Evidence  │
        │tools/    │  │Store    │  │Base      │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            ↓
              ┌──────────────────────────┐
              │    Evidence Engine        │
              │  (确定性规则引擎，不做ML)   │
              │                            │
              │  特征值 + 统计参考 + 规则   │
              │         ↓                  │
              │   EvidenceReport /         │
              │   TaskOrientedReport       │
              └────────────────────────────┘
```

**关键设计原则**：原始 PPG 波形数据**永远不经过 LLM**。Agent 只接收 `(dataset, subject_id, seg_idx)` 三元组，所有特征数据从 parquet 缓存中读取。

---

## 二、系统输入

### 2.1 run_agent() 调用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `agent` | CompiledStateGraph | (必填) | 由 `create_ppg_agent()` 创建 |
| `dataset` | str | `"vital"` | 数据集标识: `"vital"`/`"vitaldb"` 或 `"mimic"`/`"mimic3"` |
| `subject_id` | str | `"p000001"` | 患者 ID |
| `seg_idx` | int | `0` | segment 索引（一个患者有多个 10 秒片段） |
| `query` | str \| None | `None` | 用户提问，None 时使用默认问题 |

### 2.2 拼装后的 LLM 输入

```python
# agent.py:86-93 拼装的实际输入
input_msg = (
    f"{query}\n\n"
    f"Inputs:\n"
    f"- dataset: {dataset}\n"
    f"- subject_id: {subject_id}\n"
    f"- seg_idx: {seg_idx}\n"
)
```

Agent 收到的 messages：
```json
[
  {"role": "system", "content": "<SYSTEM_PROMPT>"},
  {"role": "user", "content": "评估 p000160 第5个segment的血压风险\n\nInputs:\n- dataset: vital\n- subject_id: p000160\n- seg_idx: 5"}
]
```

### 2.3 默认 query

```python
"Does this segment show a pattern consistent with elevated SBP? Assess reliability and uncertainty."
```

---

## 三、系统输出

### 3.1 run_agent() 返回结构

```python
{
    "output": str,              # LLM 最终文本回复（提取的最后一条 AI message）
    "context": PPGContext,      # 全局上下文对象（当前工具层基本未使用）
    "messages": list,           # LangGraph 完整消息历史
    "full_tool_outputs": dict,  # 工具完整输出追踪（用于调试）
}
```

### 3.2 中间输出——EvidenceReport（T2 核心报告）

```python
EvidenceReport:
    signal: SignalRef              # {dataset, subject_id, seg_idx, sbp_label, dbp_label}
    quality: QualitySummary        # {quality_tier, quality_score, coverage, valid_beat_ratio, quality_flags}
    observations: list[dict]       # 各特征的值、percentile level
    supporting_evidence: list[EvidenceItem]    # 支持目标假设的证据
    contradicting_evidence: list[EvidenceItem] # 反对/非特异性证据
    statistical_context: dict      # {matched_pattern, condition_stats, feature_percentiles, segment_reference_labels}
    alternative_explanations: list[str]
    confidence: "low" | "medium"   # 最高 medium，不允许 high
    uncertainty: list[str]         # 不确定性说明
    diagnostic_safety_note: str    # 诊断安全声明
    direct_answer: str             # 一句话结论
```

### 3.3 中间输出——EvidenceItem（单条证据）

```python
EvidenceItem:
    feature: str           # 特征名，如 "reflection_index_mean"
    value: float | None    # 特征值
    level: str             # percentile 等级: above_p95/above_p85/within_iqr/below_p20/below_p05/unknown
    direction: str         # 方向描述，如 "consistent_with_increased_wave_reflection"
    strength: "weak" | "moderate"
    rationale: str         # 来自 feature_rules.json 的解释文本
    reference: dict        # 参考统计 {n, p20, p25, p50, p75, p85, p95}
```

### 3.4 中间输出——TaskOrientedReport（T1-T6 报告）

```python
TaskOrientedReport:
    signal: SignalRef
    report_type: "quality_report" | "evidence_report" | "comparison_report"
    task_family: str                # T1~T6 或 comparison
    direct_answer: str              # 一句话结论
    quality_summary: QualitySummary
    key_observations: list[dict]
    supporting_evidence: list[EvidenceItem]
    contradicting_or_nonspecific_evidence: list[EvidenceItem]
    cohort_context: dict            # 队列上下文
    uncertainty: list[str]
    safety_note: str
    recommended_next_query: str     # 建议的下一步提问
    allowed_conclusion_boundary: str
    prohibited_diagnostic_claims: list[str]  # 禁止的诊断声明
```

### 3.5 中间输出——QualitySummary（质量摘要）

```python
QualitySummary:
    quality_tier: str | None   # "good" | "fair" | "poor"
    quality_score: float       # 0~1
    coverage: float            # 信号覆盖率 0~1
    valid_beat_ratio: float    # 有效beat占比 0~1
    quality_flags: list[str]   # 触发的质量标记
```

### 3.6 质量标记（quality_flags）触发条件

```python
# evidence_engine.py:697-722
# 以下任一条件触发对应 flag：
quality_tier != "good"        → "quality_tier={value}"
quality_score < 0.75          → "quality_score_below_0.75"
coverage < 0.75               → "coverage_below_0.75"
valid_beat_ratio < 0.80       → "valid_beat_ratio_below_0.80"
```

---

## 四、7 类分析场景（Task Families）

### 4.1 T1 — 信号质量与可靠性评估

| 项目 | 内容 |
|------|------|
| **工具** | `assess_signal_reliability` |
| **输入** | dataset, subject_id, seg_idx |
| **核心逻辑** | 检查 quality_flags 是否为空；有 flag → 限制可靠性，无 flag → 可解读 |
| **输出报告** | `report_type="quality_report"`, `task_family="T1_signal_quality_reliability"` |
| **direct_answer 示例** | "The segment is interpretable for cautious segment-level evidence." |
| **下一步建议** | 如果可靠性受限，检查相邻 segments |

### 4.2 T2 — 高 SBP 方向性模式筛查（核心功能）

| 项目 | 内容 |
|------|------|
| **工具** | `screen_elevated_sbp_pattern`, `generate_evidence_report` |
| **输入** | dataset, subject_id, seg_idx |
| **核心逻辑** | 检查 4 个核心特征（RI/LASI/PTT/灌注指数），对每个特征：<br>1. 查特征值在队列中的 percentile 等级<br>2. 根据 feature_rules.json 判断方向（supports_elevated_sbp_when: "high" 或 "low"）<br>3. RI↑ + LASI↑ + PTT↓ → 支持证据；反之 → 矛盾证据<br>4. 匹配合并模式（`_pattern()`: ri_high / lasi_high / ptt_short / ri_high_lasi_high_and_ptt_short 等）<br>5. 查该模式下的 condition_stats（BP 分布统计）<br>6. 确定 confidence: ≥2条支持 + 质量OK + ​队列n≥1000 → medium, 否则 low |
| **输出报告** | EvidenceReport (完整) 或 TaskOrientedReport (task_family="T2_elevated_sbp_directional_pattern") |
| **置信度规则** | low（默认）/ medium（需满足：无质量标记 + ≥2条支持证据 + 条件统计n≥1000） |

**T2 方向性判断矩阵：**

| 特征 | 规则(supports_elevated_sbp_when) | above_p85/p95 时 | below_p20/p05 时 | 其他 |
|------|------|------|------|------|
| RI | "high" | → 支持（波反射增强） | → 矛盾 | 不特异 |
| LASI | "high" | → 支持（动脉硬度代理） | → 矛盾 | 不特异 |
| PTT | "low" | → 矛盾 | → 支持（传输时间缩短） | 不特异 |
| 灌注指数 | "none" | 不用于 SBP | → 矛盾（可能灌注/传感器问题） | 不特异 |

### 4.3 T3 — 血管硬度/波反射代理筛查

| 项目 | 内容 |
|------|------|
| **工具** | `screen_vascular_stiffness_pattern` |
| **输入** | dataset, subject_id, seg_idx |
| **核心逻辑** | 检查 RI/LASI/PTT 3 个特征：<br>RI↑ 或 LASI↑ → 支持硬度代理<br>PTT↓ → 支持（更短的传输时间）<br>特征在正常范围 → 非特异性 |
| **输出报告** | `task_family="T3_vascular_stiffness_wave_reflection_proxy"` |
| **判定条件** | ≥2条支持证据 → 有方向性证据；<2条 → 无清晰模式 |

### 4.4 T4 — PTT/BP/形态学一致性解释

| 项目 | 内容 |
|------|------|
| **工具** | `explain_ptt_bp_consistency` |
| **输入** | dataset, subject_id, seg_idx |
| **核心逻辑** | 1. 读取参考 SBP 标签，以 130mmHg 为阈值判断高/正常<br>2. 检查 PTT 方向：PTT↓ + SBP高 → 一致；PTT↑ + SBP正常 → 一致<br>3. 检查 RI/LASI 形态学代理与 SBP 的一致性 |
| **输出报告** | `task_family="T4_ptt_bp_morphology_consistency"` |

### 4.5 T5 — 灌注/传感器/外周问题筛查

| 项目 | 内容 |
|------|------|
| **工具** | `explain_perfusion_or_sensor_issue` |
| **输入** | dataset, subject_id, seg_idx |
| **核心逻辑** | 1. 灌注指数低于 p20/p05 → 可能低灌注/传感器耦合问题<br>2. FSQI > 0 → 可能存在信号质量变化<br>3. quality_flags → 可能反映运动/覆盖/预处理问题 |
| **输出报告** | `task_family="T5_perfusion_sensor_peripheral_issue"` |

### 4.6 T6 — Segment 可解读性分类

| 项目 | 内容 |
|------|------|
| **工具** | `classify_segment_interpretability` |
| **输入** | dataset, subject_id, seg_idx |
| **核心逻辑** | 综合 T2(elevated SBP) + T5(perfusion) 结果：<br>1. 无质量标记 + ≥3个核心特征可用 → 可解读<br>2. 无质量标记 + T2和T5都无发现 → 可解读但正常/非特异<br>3. 有质量标记 → 质量受限 |
| **输出报告** | `task_family="T6_normal_or_nonspecific_segment"` |

### 4.7 Comparison — 多任务家族对比

| 项目 | 内容 |
|------|------|
| **工具** | `generate_task_oriented_report(task_family="comparison")` |
| **输入** | dataset, subject_id, seg_idx |
| **核心逻辑** | 同时运行 T2 + T3 + T5，汇总对比 |
| **输出报告** | `task_family="comparison_task_family_summary"` |

---

## 五、14 个工具详解

### 5.1 数据获取类

#### `load_segment`
```
功能: 加载一个 segment 的缓存元数据和核心特征摘要
输入: dataset (str), subject_id (str), seg_idx (int)
处理: PPGDataStore.load_segment_features() → 读取 segments.parquet
输出: {
    "status": "success",
    "subject_id": "p000001",
    "seg_idx": 0,
    "dataset": "vital",
    "age": 62, "gender": "M",
    "quality_tier": "good", "quality_score": 0.92, "coverage": 0.95,
    "n_beats_ppg": 12, "n_beats_ecg": 12,
    "sbp_label": 138.5, "dbp_label": 72.3,
    "reflection_index_mean": 0.52, "lasi_mean": 15.6,
    "ptt_mean_ms": 385.2, "perfusion_index_mean": 2.3,
    "message": "Loaded cached segment features; raw waveforms were not passed to the LLM."
}
返回字段: SEGMENT_SUMMARY_COLUMNS 中存在的所有字段（14个列）
```

#### `get_segment_features`
```
功能: 返回核心形态学、PTT、ABP脉压特征
输入: dataset, subject_id, seg_idx, feature_group="core"
处理: PPGDataStore.load_segment_features() → 筛选 CORE_SEGMENT_FEATURES (15个)
输出: {
    "status": "success",
    "dataset": "vital", "subject_id": "p000001", "seg_idx": 0,
    "feature_group": "core",
    "features": {
        "reflection_index_mean": 0.52, "reflection_index_std": 0.08,
        "lasi_mean": 15.6, "lasi_std": 3.2,
        "pulse_width_50_mean": 0.21, "pulse_width_90_mean": 0.35,
        "perfusion_index_mean": 2.3,
        "skewness_mean": 0.15, "kurtosis_mean": 3.8,
        "spectral_entropy_mean": 2.1, "wavelet_entropy_mean": 1.8,
        "signal_quality_fsqi_mean": 0.05,
        "ptt_mean_ms": 385.2, "ptt_std_ms": 15.3,
        "abp_pp_mean": 66.2
    }
}
特征列表 (15个):
  形态学: reflection_index(mean/std), lasi(mean/std), pulse_width_50, pulse_width_90,
          skewness, kurtosis
  灌注:   perfusion_index
  频域:   spectral_entropy, wavelet_entropy
  质量:   signal_quality_fsqi
  时间:   ptt(mean/std)
  血压:   abp_pp
```

#### `get_beat_summary_from_cache`
```
功能: 逐 beat 特征一致性摘要（mean/std/median/min/max）
输入: dataset, subject_id, seg_idx
处理: PPGDataStore.load_beat_features() → 读取 beats.parquet → 按特征统计
输出: {
    "status": "success",
    "dataset": "vital", "subject_id": "p000001", "seg_idx": 0,
    "beat_rows": 12,
    "valid_beat_ratio": 0.92,
    "features": {
        "reflection_index": {
            "mean": 0.52, "std": 0.08, "median": 0.51, "min": 0.38, "max": 0.68
        },
        "lasi": {...}, "pulse_width_50": {...}, "perfusion_index": {...},
        "skewness": {...}, "kurtosis": {...},
        "spectral_entropy": {...}, "signal_quality_fsqi": {...},
        "period_seconds": {...}
    }
}
注意: 当 beats.parquet 中无数据时返回 status="warning"
```

### 5.2 质量评估类

#### `get_segment_quality`
```
功能: 返回质量等级、评分、覆盖率、有效beat比例、质量标记
输入: dataset, subject_id, seg_idx
处理: EvidenceEngine._quality_summary() → 计算 QualitySummary
输出: {
    "status": "success",
    "quality_tier": "good",           # good / fair / poor
    "quality_score": 0.92,            # 0~1
    "coverage": 0.95,                 # 0~1
    "valid_beat_ratio": 0.92,         # 0~1
    "quality_flags": [],              # 触发的标记列表，如 ["coverage_below_0.75"]
    "beat_rows": 12
}
质量标记生成规则:
  - quality_tier != "good"      → "quality_tier=fair" (或 poor)
  - quality_score < 0.75        → "quality_score_below_0.75"
  - coverage < 0.75             → "coverage_below_0.75"
  - valid_beat_ratio < 0.80     → "valid_beat_ratio_below_0.80"
```

### 5.3 证据筛查类（核心业务工具）

#### `assess_signal_reliability` → T1
```
功能: 评估信号是否可靠到能做形态学/PTT解读
输入: dataset, subject_id, seg_idx
处理: EvidenceEngine.assess_signal_reliability()
输出: {
    "status": "success",
    "report": { TaskOrientedReport(T1) 完整字典 }
}
核心判断:
  - 无 quality_flags → 可解读，quality支持证据
  - 有 quality_flags → 每个flag生成一条限制性证据
```

#### `screen_elevated_sbp_pattern` → T2（最核心）
```
功能: 筛查一个 segment 是否方向性一致地指向高 SBP 模式
输入: dataset, subject_id, seg_idx
处理: EvidenceEngine.screen_elevated_sbp_pattern()
输出: {
    "status": "success",
    "direct_answer": "The segment shows low-confidence directional consistency...",
    "confidence": "low",
    "quality": { QualitySummary },
    "supporting_evidence": [ EvidenceItem, ... ],
    "contradicting_evidence": [ EvidenceItem, ... ],
    "statistical_context": {
        "matched_pattern": "ri_high",     # 匹配到的特征模式
        "condition_stats": {...},          # 该模式下的BP分布
        "feature_percentiles": {...},      # 各特征的percentile
        "segment_reference_labels": {...}  # SBP/DBP标签
    },
    "uncertainty": [ ... ],
    "diagnostic_safety_note": "..."
}
直接回答生成逻辑:
  - 有支持证据 + 低质量 → "low-confidence directional consistency..."
  - 有支持证据 + 好质量 → "medium-confidence directional consistency..."
  - 无支持证据 → "does not show a clear elevated-SBP-related pattern"
  - 有质量标记 → 追加 "Signal quality limitations..."
  - 有矛盾证据 → 追加 "Contradicting or nonspecific evidence..."
```

#### `screen_vascular_stiffness_pattern` → T3
```
功能: 筛查 RI/LASI/PTT 是否指向血管硬度或波反射代理
输入: dataset, subject_id, seg_idx
输出: { "status": "success", "report": { TaskOrientedReport(T3) } }
```

#### `explain_ptt_bp_consistency` → T4
```
功能: PTT 方向是否与参考 BP 标签一致
输入: dataset, subject_id, seg_idx
输出: { "status": "success", "report": { TaskOrientedReport(T4) } }
```

#### `explain_perfusion_or_sensor_issue` → T5
```
功能: 筛查低灌注/传感器/外周问题
输入: dataset, subject_id, seg_idx
输出: { "status": "success", "report": { TaskOrientedReport(T5) } }
```

#### `classify_segment_interpretability` → T6
```
功能: 综合分类 segment 的可解读性
输入: dataset, subject_id, seg_idx
输出: { "status": "success", "report": { TaskOrientedReport(T6) } }
```

### 5.4 统计查询类

#### `query_feature_reference`
```
功能: 查询一个特征在队列中的参考百分位数和与 BP 的相关性
输入: feature (str), dataset="vital", quality_filter="quality_ok"
处理: EvidenceBase.reference_stats() + EvidenceBase.correlation()
输出: {
    "status": "success",
    "feature": "reflection_index_mean",
    "dataset": "vital",
    "quality_filter": "quality_ok",
    "reference": {
        "n": 12345, "mean": 0.42, "std": 0.19,
        "p05": 0.127, "p20": 0.258, "p25": 0.285,
        "p50": 0.401, "p75": 0.537, "p85": 0.627, "p95": 0.765
    },
    "sbp_correlation": {"spearman_r": 0.037, "p_value": 0.0, "n": 3267152},
    "dbp_correlation": {"spearman_r": 0.090, "p_value": 0.0, "n": 3267152}
}
```

#### `query_condition_stats`
```
功能: 查询预定义特征模式下的队列血压分布
输入: pattern (str), dataset="vital", quality_filter="quality_ok"
处理: EvidenceBase.condition_stats()
pattern 可选值: "ri_high", "lasi_high", "ptt_short",
                "ri_high_lasi_high_and_ptt_short", "ri_high_and_ptt_short",
                "lasi_high_and_ptt_short"
输出: {
    "status": "success",
    "pattern": "ri_high",
    "dataset": "vital",
    "quality_filter": "quality_ok",
    "condition_stats": {
        "n": 489655,
        "sbp_median": 132.75, "sbp_mean": 132.88,
        "sbp_iqr": [117.55, 147.42],
        "sbp_gt_140_ratio": 0.37, "sbp_gt_130_ratio": 0.55,
        "dbp_median": 65.30, "dbp_gt_90_ratio": 0.039
    }
}
```

### 5.5 报告生成类

#### `generate_evidence_report`
```
功能: 生成完整的结构化 EvidenceReport（等同于 T2 的完整输出）
输入: dataset, subject_id, seg_idx
输出: { "status": "success", "report": { EvidenceReport 完整字典 } }
```

#### `generate_task_oriented_report`
```
功能: 生成 E.5 任务导向报告模板，可按 task_family 或 comparison 指定
输入: dataset, subject_id, seg_idx, task_family="T2", report_type=None
task_family 别名:
  "1"/"t1"/"quality"/"reliability"      → T1
  "2"/"t2"/"elevated_sbp"/"sbp"        → T2 (默认)
  "3"/"t3"/"stiffness"/"vascular_stiffness" → T3
  "4"/"t4"/"ptt"/"ptt_bp"              → T4
  "5"/"t5"/"perfusion"/"sensor"        → T5
  "6"/"t6"/"normal"/"nonspecific"      → T6
  "comparison"/"comparison_report"      → comparison
report_type="comparison_report"          → comparison (与 task_family 等效)
输出: { "status": "success", "report": { TaskOrientedReport 完整字典 } }
```

---

## 六、Pipeline 完整流程（以 T2 为例）

```
用户提问: "评估 p000160 第5个segment的血压风险"

Step 1: Agent 收到 user message
  → LLM 决定调用 load_segment("vital", "p000160", 5)
  → 返回: 14个核心元数据字段

Step 2: LLM 决定调用 get_segment_quality("vital", "p000160", 5)
  → PPGDataStore 读 segments.parquet + beats.parquet
  → EvidenceEngine._quality_summary() 计算
  → 返回: quality_tier="good", score=0.92, coverage=0.95, flags=[]

Step 3: LLM 决定调用 screen_elevated_sbp_pattern("vital", "p000160", 5)
  → EvidenceEngine.screen_elevated_sbp_pattern():
    ├─ 加载 segment + beats
    ├─ 计算 QualitySummary
    ├─ 对 RI/LASI/PTT/灌注 4个特征:
    │   ├─ 查 EvidenceBase.reference_stats() → percentile等级
    │   ├─ 查 EvidenceBase.feature_rule() → 方向规则
    │   └─ 判断: above_p85? → 支持 / below_p20? → 矛盾 / 其他 → 不特异
    ├─ _pattern(): 合并特征模式 "ri_high_lasi_high_and_ptt_short"
    ├─ _statistical_context(): 查 condition_stats("ri_high_lasi_high_and_ptt_short")
    │   → "VitalDB中同时满足RI↑+LASI↑+PTT↓的segment, SBP>130 占 58%"
    ├─ _confidence(): ≥2支持 + 无质量标记 + n≥1000 → "medium"
    └─ _direct_answer(): 拼接最终答案
  → 返回: EvidenceReport

Step 4: LLM 综合以上信息，生成最终解读文本
  → run_agent() 提取最后一条 AI message 作为 output
```

---

## 七、知识层详解

### 7.1 EvidenceBase 数据源

| 文件 | 内容 | 用途 |
|------|------|------|
| `pulse_db_reference_stats.json` | 所有特征 × 数据集 × quality_filter 的百分位分布 | 判断特征值是"偏高"还是"偏低" |
| `pulse_db_correlations.json` | 特征 × BP标签的 Spearman 相关系数 | 提供特征与BP的统计关联上下文 |
| `pulse_db_condition_stats.json` | 预定义模式下的 BP 分布（中位数/IQR/>140比例等） | 为模式匹配提供队列背景 |
| `feature_rules.json` | 5个特征的手写方向规则 + 6个task_family的边界约束 | 解释特征方向的生理意义 |

### 7.2 Percentile 等级分类

```python
# evidence_base.py:83-99
value >= p95  → "above_p95"
value >= p85  → "above_p85"
value >= p75  → "above_p75"
value <= p05  → "below_p05"
value <= p20  → "below_p20"
value <= p25  → "below_p25"
其他          → "within_iqr"
```

---

## 八、数据层详解

### 8.1 PPGDataStore 数据源

| 数据文件 | 内容 |
|----------|------|
| `{data_root}/vitaldb/features/segments.parquet` | segment 级特征（每行一个 10 秒片段） |
| `{data_root}/vitaldb/features/beats.parquet` | beat 级特征（每行一个 heartbeat） |
| `{data_root}/vitaldb/labels.csv` | 受试者疾病标签 |
| `{data_root}/mimic3/features/segments.parquet` | MIMIC segment 级特征 |
| `{data_root}/mimic3/features/beats.parquet` | MIMIC beat 级特征 |
| `{data_root}/mimic3/labels.csv` | MIMIC 受试者疾病标签 |

### 8.2 数据查询方式

```python
store.load_segment_features(dataset, subject_id, seg_idx)  → dict  # 单segment
store.load_beat_features(dataset, subject_id, seg_idx)     → list[dict]  # 多beat
store.load_subject_segments(dataset, subject_id)            → DataFrame  # 单受试者所有segment
store.query_segments(dataset, columns)                      → DataFrame  # 批量查询
store.load_labels(dataset, subject_id)                      → dict  # 疾病标签
```

### 8.3 PPGContext（预留但当前未充分利用）

```python
PPGContext:  # context.py
    ppg_raw: np.ndarray        # 原始PPG信号（当前工具未填充）
    abp: np.ndarray            # ABP参考信号
    ecg: np.ndarray            # ECG信号
    beats: list[np.ndarray]    # 逐beat波形
    beat_features: dict        # 逐beat特征
    deep_features: dict        # 深入分析结果
    risk_flags: list[dict]     # 风险标记
    analysis_log: list[str]    # 分析路径日志
```

---

## 九、Middleware 层

### QwenToolCallMiddleware

```
功能: 拦截 Qwen2.5 模型的 <tool_call> XML 输出，转换为 LangChain 标准 ToolCall 格式

原因: Qwen2.5-14B 通过 vLLM 返回原生 <tool_call> XML:
  <tool_call>
  {"name": "load_segment", "arguments": {"dataset": "vital", ...}}
  </tool_call>

  而 LangChain 期望 OpenAI 格式的结构化 tool_calls 字段。

处理: 正则提取 <tool_call> 块 → JSON解析 → 构造 AIMessage(tool_calls=[...])
      同时从 content 中清除 XML 块，保留干净文本。
```

---

## 十、System Prompt 规定的默认工作流

```
1. load_segment          → 获取基础信息
2. get_segment_quality   → 评估信号质量
3. get_segment_features  → 获取详细特征
4. 调用匹配用户意图的 task-family 工具  → 核心分析
5. query_feature_reference / query_condition_stats → 需要时查询队列上下文
6. generate_task_oriented_report → 生成最终综合报告
```

---

## 十一、安全约束

- 最高 confidence 为 **medium**（禁止 high）
- 禁止的诊断声明：诊断高血压、诊断动脉硬化、证明血管疾病、排除心血管疾病
- 使用 "directionally consistent with" 而非 "proves" 或 "diagnoses"
- 必须提及信号质量限制和备选解释
- 单特征家族支持时 confidence 必须为 low
