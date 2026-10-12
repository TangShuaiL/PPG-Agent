# PPG-Agent 多智能体架构图

> 生成日期：2026-08-18；Agent 2 分层比较引擎重构后更新：2026-08-21
> 用途：汇报 PPT 架构图素材（Mermaid 源码，可渲染后截图或直接重绘）
> 依据代码：`ppg_agent/agents/{supervisor,signal_processor_agent,cardiovascular_agent,adaptation_agent}.py`

---

## 图 1：总体架构（含子 Agent 内部结构与共享基础设施）

```mermaid
flowchart TB
    USER["用户 / Web 前端<br/>自然语言 query"]

    %% ════════════════ Supervisor ════════════════
    subgraph SUP["Supervisor · 中央编排器（LLM）"]
        direction TB
        S1["① 意图解析 → 结构化计划<br/><b>SupervisorPlan</b> = {thought, needs_dispatch,<br/>task_queue: SubTask[ ], direct_reply}<br/><b>SubTask</b> = {target_agent, task, params}"]
        S2["② 路由分发<br/>signal_processing / cardiovascular / ml_prediction"]
        S3["③ 综合报告<br/>一句话结论 + 2-3 个关键指标 + 下一步建议<br/>（≤4 句，禁止复述过程）"]
        S1 --> S2 --> S3
    end

    %% ════════════════ Agent 1 ════════════════
    subgraph A1["Agent 1 · signal_processor（LangGraph 状态机）"]
        direction TB
        A1N["确定性管线节点<br/>load → normalize → scan<br/>→ segment_beats（onset 修正）<br/>→ extract_features"]
        A1E["signal_engine 确定性引擎<br/>processing_params：14 参数真源<br/>（A 类 7 可调 + B 类 5 锁定 + 2 开关）"]
        A1L["LLM 决策节点<br/>tools: visualize_signal / assess_quality<br/>（只读摘要，不碰原始波形）"]
        A1N --- A1E
        A1N --> A1L
    end

    subgraph AD["Agent 1 · adapt 分支 AdaptationSession（ReAct）"]
        direction TB
        AD1["diagnose → propose<br/><b>ProposeOptions</b> = [ParamChange<br/>{param, value, user_desc}]"]
        AD2["apply（用户点选 A/B/C…）<br/>→ validate 批量验证"]
        AD3["验收门禁<br/>分半信度中位 ρ≥0.5<br/>且符号对齐 VAI ρ≥0.25"]
        AD4["finalize 保存 preset / revert 回退"]
        AD1 --> AD2 --> AD3 --> AD4
    end

    %% ════════════════ Agent 2（v3 重构：纵向个人比较） ════════════════
    subgraph A2["Agent 2 · cardiovascular（LangGraph 状态机）"]
        direction TB
        A2A["assess 节点（确定性）<br/>personal_engine.compare_segment：<br/>双参照（长期基线+近 7 段）三分类 / beat 层离群率 / 漂移"]
        A2R["<b>CompareReport</b> = {verdict, n_history, features:<br/>{Δ, 阈值(MDC/个人 3σ-MAD), verdict_long/recent},<br/>beat_layer, drift, VOLATILE 标记}"]
        A2L["interpret LLM 节点（只决策，不叙事）<br/>tools: personal_trend / bp_context /<br/>beat_bp_coupling / submit_findings（≤8 轮）"]
        A2F["<b>Findings</b> = {changed_features: [引擎判定特征<br/>Δ/阈值逐字取自引擎], stable_note, bp_note,<br/>follow_up_suggestion}"]
        A2A --> A2R --> A2L --> A2F
    end

    %% ════════════════ Agent 3 ════════════════
    A3["Agent 3 · ml_prediction<br/>beat 级特征预训练（规划中）<br/>未就绪时如实告知，不编造预测"]

    %% ════════════════ 共享基础设施 ════════════════
    CTX["<b>PPGContext</b> · 请求级共享状态（contextvars 隔离）<br/>signal / beats / keypoints / features / age<br/>原则：原始波形永不进 LLM，只传统计摘要"]
    DB[("SQLite ppg_phenotype.db<br/>subjects / segments / segment_features<br/>baselines / vai_age_stratified")]
    PQ[("*_beats_full.parquet 宽表<br/>逐 beat 特征 + 内联 ABP 血压<br/>（personal_engine 直读）")]
    KNOW[("knowledge/ 先验知识库<br/>feature_rules（40 条）· mdc_reference · feature_icc<br/>bp_baseline_by_condition · rho_bp_min_segments ·<br/>beat_bp_coupling_reference")]

    %% ════════════════ 主数据流 ════════════════
    USER -->|"自然语言 + <b>SignalRef</b>{source, subject_id, seg_idx, fs}"| SUP
    S2 -->|"SubTask(signal_processing)"| A1
    S2 -->|"SubTask(cardiovascular)"| A2
    S2 -.->|"SubTask(ml_prediction)"| A3
    A1 -->|"数据集适配意图（task=dataset_adaptation）"| AD

    A1 <-->|"读/写"| CTX
    A2 <--->|"读 beat_summary 交接面<br/>写 agent2_identity"| CTX

    A1 -->|"beat_summary()（beat 数/有效率/beat 级 BP）<br/>+ 逐 beat 记录，经 PPGContext 交接"| A2

    A1 -->|"ctx.<b>to_summary()</b> = {signal_ref, stage,<br/>num_beats, quality_tier, …}"| S3
    A2 -->|"Findings（结构化发现）+ CompareReport（确定性证据）"| S3

    DB -->|"波形路径 + 元数据"| A1
    PQ -->|"段/beat 级特征与血压历史"| A2A
    KNOW -->|"retrieve_knowledge(features)<br/>综合阶段注入生理规则"| S3
    KNOW -->|"MDC 阈值 / 疾病分层 BP 基线（仅正推）/ ρ 门槛"| A2L

    S3 -->|"自然语言报告（含 safety note）"| USER
```

---

## 图 2：一次完整分析的结构化消息时序

```mermaid
sequenceDiagram
    participant U as 用户 / Web
    participant S as Supervisor
    participant A1 as Agent 1 信号处理
    participant A2 as Agent 2 纵向比较
    participant E as personal_engine + parquet 宽表

    U->>S: 自然语言 query + SignalRef{source, subject_id, seg_idx}
    S->>S: 意图解析 → SupervisorPlan{thought, needs_dispatch, task_queue}

    S->>A1: SubTask{target_agent=signal_processing, task, params}
    A1->>A1: load → normalize → segment_beats → extract_features（确定性）
    A1-->>S: ctx.to_summary(){signal_ref, stage, num_beats, quality_tier}

    S->>A2: SubTask{target_agent=cardiovascular}（携带受试者身份 config）
    A2->>E: compare_segment(ds, sid, k)：历史=seg 0~k−1，当前段永不进历史
    E-->>A2: CompareReport{verdict, 双参照判定, beat_layer, drift}
    A2->>A2: interpret LLM：按需 personal_trend / bp_context / beat_bp_coupling → submit_findings
    A2-->>S: Findings{changed_features(引擎事实 only), stable_note, bp_note, follow_up}

    S->>S: retrieve_knowledge(features) 注入生理规则；纵向解读纪律约束措辞
    S-->>U: 综合报告：本人历史变化 → beat 证据 → 人群位置 → 标签语境 → 安全声明
```

---

## 结构化消息速查表（PPT 标注用）

| 消息 | 产生方 → 消费方 | 内容 |
|------|----------------|------|
| `SignalRef` | 用户 → Supervisor | `{source, subject_id, seg_idx, fs, duration_sec}`，全流程唯一入口 |
| `SupervisorPlan` | Supervisor 内部 | `{thought, needs_dispatch, task_queue, direct_reply}` |
| `SubTask` | Supervisor → 子 Agent | `{target_agent, task(人类可读描述), params(路由参数)}` |
| `to_summary()` | Agent 1 → Supervisor | 上下文摘要：阶段状态、beat 数、质量分层——**不含波形数据** |
| `beat_summary()` | Agent 1 → Agent 2 | 当前段 beat 数/有效率/beat 级 BP 中位，经 PPGContext 交接，与引擎 beat 层交叉校验 |
| `CompareReport` | Agent 2 内部（personal_engine 输出） | verdict 门控（ok/no_history/insufficient_history/低质量）、双参照三分类、beat 层离群率、漂移、VOLATILE 标记 |
| `Findings` | Agent 2 → Supervisor | 引擎判 real_change 的特征（Δ/阈值/方向逐字取自引擎）+ 稳定性注记 + 血压语境 + 随访建议；Agent 2 不产散文，措辞归 Supervisor |
| `ProposeOptions` | 适配会话 → 用户 | `[ParamChange{param, value, user_desc}]`，规则生成 + LLM 兜底 |
| 知识注入 | knowledge/ → Supervisor | `retrieve_knowledge(features)`：特征→生理方向的规则文本 |

## Agent 2 能力证据状态（纵向比较引擎，test 集滑动游走验证，2026-08）

范式：train 拟合人群先验（knowledge/ JSON）→ test 集滑动游走验证（vital 429 人全量 + mimic 抽 120 人，对数间隔评估点，复现运行时同款语义）。
协议与全部数值见 `analysis/10_heldout_validation/02_longitudinal_validate.py`（`output/longitudinal_summary.csv` + `longitudinal_report.json`）。

| 能力 | 证据状态 | held-out 关键数值 | 处置 |
|------|---------|------------------|------|
| 变化判定（个人 3σ-MAD/MDC 双参照） | ✅ test-validated | 中位误报率 vital 0.058 / mimic 0.026（250 万次判定）；但 8 特征 FP>10%（kurtosis 0.354 等） | 保留；8 高波动特征标 VOLATILE 叙事降级 |
| beat 层离群率 | ✅ test-validated | 中位 0.0（两库，3σ 围栏特异优先） | 保留 |
| beat×BP 耦合 | ✅ test-validated | 相邻段符号一致率 vital 0.68–0.79 / mimic 0.70–0.80 | 保留（体系最稳信号，去混杂版为权威） |
| 百分位校准（人群位置） | ✅ vital / ⚠️ mimic | frac_above_p50：vital 0.4835 达标；mimic 0.5676 略超上界（ICU 人群漂移，记录在案） | 仅 ICC≥0.6 特征输出，其余 suppressed |
| 个人内 ρ(feature,BP) | ⚠️ 弱参考 | split-half 符号一致率 0.64–0.70 未达 0.70 | 门槛 ≥20 段；只能以「弱相关」措辞引用 |
| 疾病×BP 分层基线 | ⚠️ 仅正推 | 受试者中位口径 mimic 分离 11.9mmHg 达标；vital 2.9（术前降压药控制的临床现实） | 已知标签 → 该组 BP 分布的正推描述；严禁反推疾病 |
| 热身门槛（历史长度敏感性） | ℹ️ 阴性发现 | FP 对历史长度几乎平坦（3→50 段：vital 0.073→0.077） | compare ≥3 段门槛维持，ρ 类统计坚持 ≥20 段 |
| VAI / 表型亚型 / 疾病关联 / interpretation_rules / window_assess | ❌ 已证伪或降级 | 疾病关联方向复现率≈随机；VAI 等纵向场景不适用 | 已从 Agent 2 与前端展示移除（compute_vai 保留供 analysis/ 复现脚本；百分位等描述性报告路径保留） |

## 三条贯穿性设计原则（图注脚）

1. **LLM 只做调度与表达**：所有数值计算走确定性引擎（signal_engine / personal_engine），Agent 2 只决策分析路径，叙事归 Supervisor
2. **原始波形永不进 LLM**：跨 Agent 只传结构化摘要与特征统计量
3. **状态请求级隔离**：PPGContext 基于 contextvars，并发分析互不污染；跨节点→工具的状态走 RunnableConfig.configurable（ToolNode 线程池不传播 contextvars）
