"""Supervisor Agent — central router with NO domain tools.

Three-phase orchestration (parse → dispatch → synthesis):

    (1) intent parse — structured plan: thought / needs_dispatch / task_queue
    (2) dispatch     — route sub-tasks to the sub-agents' deterministic engines
    (3) synthesis    — combine sub-agent structured results + retrieved knowledge
                       into a concise natural-language report (the chat reply)

Design constraints:
    * The Supervisor holds no signal-processing / cardiovascular tools — it is
      a pure reasoning layer.  Sub-agent capabilities are registered as tool
      packs in the chat router; the LLM (DeepSeek) appears only here.
    * The synthesis phase is where the "supervisor adds value": it cross-links
      independent evidence (e.g. a real longitudinal change AND a same-direction
      model-attribution keypoint → two views corroborating, never inferring one
      from the other) and compresses the reply
      to a few sentences — details live in the frontend cards, not in prose.
    * retrieve_knowledge() is the knowledge hook for synthesis.  It currently
      reads the existing RuleEngine (feature → physiological direction rules);
      a full cardiovascular knowledge base / RAG plugs in here later without
      touching the orchestration.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured routing protocol
# ---------------------------------------------------------------------------

TargetAgent = Literal[
    "signal_processing",   # Agent 1: beats, keypoints, quality, dataset adaptation
    "cardiovascular",      # Agent 2: longitudinal compare, population position, feature meaning
    "ml_prediction",       # Agent 3: deep-learning risk models (training)
]

# Canonical routing identifiers — the single source of truth.  These MUST equal
# the Literal members above; every dispatch/comparison site references these
# constants instead of bare strings, so a rename lands in exactly one place.
AGENT_SIGNAL = "signal_processing"
AGENT_CARDIO = "cardiovascular"
AGENT_ML = "ml_prediction"
TARGET_AGENTS = (AGENT_SIGNAL, AGENT_CARDIO, AGENT_ML)

# Routing domain → LLM registry key (models/model_registry.yaml).  Agent 3 is a
# deterministic BeatMAE engine with no LLM, hence absent here.  Note the
# historical skew: domain "signal_processing" ↔ registry key "signal_processor".
AGENT_LLM_KEY: dict[str, str] = {
    AGENT_SIGNAL: "signal_processor",
    AGENT_CARDIO: "cardiovascular",
}


class SubTask(BaseModel):
    """One routed sub-task for a sub-agent."""

    target_agent: TargetAgent
    task: str = Field(description="人类可读的任务描述")
    params: dict[str, Any] = Field(default_factory=dict,
                                   description="source/subject_id/seg_id 等路由参数")


class SupervisorPlan(BaseModel):
    """Structured intent-parse output of the Supervisor."""

    thought: str = Field(description="意图分析（简短）")
    needs_dispatch: bool = Field(description="是否需要路由到子 Agent")
    task_queue: list[SubTask] = Field(default_factory=list,
                                      description="要分发的子任务列表")
    direct_reply: str | None = Field(
        default=None, description="无需路由时的直接回答（可留空）")


# ---------------------------------------------------------------------------
# System prompt — the only place the Supervisor's behavior is defined
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """你是 PPG-Agent 的主管 Agent（Supervisor）——心血管表型分析多智能体系统的中央编排器。

你的职责分三阶段：
① 意图解析：理解用户意图，分解为子任务
② 路由：将子任务分发给正确的子 Agent（子 Agent 的能力通过工具调用体现）
③ 综合报告：子 Agent 返回结构化结果后，由你整合为简洁的自然语言报告

子 Agent（通过工具调用其能力）：
  - signal_processing（Agent 1，信号处理）：beat 分割、关键点、信号质量评估、特征提取、数据集适配调参
  - cardiovascular（Agent 2，表型分析）：纵向比较（和本人历史比）、个人趋势、血压语境（分组基线只正推）、人群位置、beat-BP 耦合、标量特征生理含义解释；回答"偏高/偏低意味着什么"
  - ml_prediction（Agent 3，深度学习模型）：血压/年龄模型预测（BeatMAE 嵌入查表，确定性输出；预测为群体级统计估计，必须携带置信说明，非诊断结论）；并可对预测做可解释性叙事（三层证据：近窗注意力×当前段/历史段特征点归因）——这是模型输入依赖视角，不是个体病因
  知识层同源说明：特征点知识（waveform_keypoints）与标量特征知识（features）同源于 feature_rules.json，前者由 Agent 3 解释链消费，后者由 Agent 2 的 feature_interpretation 消费。

工具选择规则：
1. 用户想分析波形/信号质量/特征提取 → analyze_signal
2. 用户想了解纵向变化/人群位置/血压解读/特征含义 → analyze_phenotype
3. 用户想得到模型预测（高血压/心血管风险等）→ ml_predict_risk
4. 用户问"为什么出这个预测值/哪些特征点驱动/解释这个预测/归因/怎么得出" → ml_prediction 的 task="explain_prediction"（模型依赖度视角）
5. 用户问"为什么偏高/偏低、这个指标什么意思、和本人历史比、人群位置" → analyze_phenotype（生理机制视角）；两类"为什么"不得混述；一句话同时含两义时可同时入队 explain_prediction 与 analyze_phenotype，综合时分视角陈述、不得互推；子 Agent 未派发时不得叙述为"无证据可用"，应说明本轮只做了解释并引导追问
6. 用户选择了一个数据集（未选受试者/segment）并提到要对数据集做特征提取适配 → task="dataset_adaptation"（由 Agent 1 适配分支经 adapt_pipeline 语义动作推进）
7. 用户放弃适配/删除已保存的用户参数 → 适配分支 adapt_pipeline(action="cancel")
8. 工具调用失败或信息不足时，友好说明并引导选择

综合报告规则（最终回复必须遵守）：
- 先给一句话结论，再列 2-3 个关键指标（质量/有效beat/偏离项数 等），最后给下一步建议
- 跨子 Agent 结果做关联解读（如 相对本人历史出现真实变化且模型归因同向特征点 → 两视角互证、不得互推）
- 总长不超过 4 句；指标用具体数字
- 禁止：复述用户的问题、客套话（"好的/没问题"）、解释工具调用过程、复述前端已展示的表格内容

知识库：综合阶段会注入生理学规则（特征→方向映射）。引用规则时说明来源（如"按 PPG 生理规则，RI↑ 提示外周阻力升高"），不要编造规则。

安全：只讨论 PPG 心血管分析相关话题，不诊断疾病、不替代医疗建议。"""


# ---------------------------------------------------------------------------
# Knowledge hook for the synthesis phase
# ---------------------------------------------------------------------------

def retrieve_knowledge(features: list[str], limit: int = 6) -> list[str]:
    """Fetch physiological rules for key features (synthesis-phase knowledge).

    Currently backed by the existing RuleEngine (feature → physiological
    direction mappings from knowledge/rules.json).  A cardiovascular knowledge
    base / RAG replaces the backend later — the signature stays.

    Args:
        features: Feature names to look up.
        limit: Max rules to return (keeps the supervisor context small).

    Returns:
        Plain-language rule strings, e.g. "RI 升高→外周阻力升高；..."
    """
    if not features:
        return []
    try:
        from ppg_agent.knowledge.rules import RuleEngine

        engine = RuleEngine()
    except Exception:  # noqa: BLE001 — knowledge is optional
        return []
    lines: list[str] = []
    for f in features[: limit * 3]:
        try:
            rule = engine.get(f)
        except KeyError:
            continue
        bits = []
        if rule.high_direction:
            bits.append(f"{rule.label or f} 升高→{rule.high_direction}")
        if rule.low_direction:
            bits.append(f"{rule.label or f} 降低→{rule.low_direction}")
        if rule.rationale:
            bits.append(rule.rationale)
        if bits:
            lines.append("；".join(bits))
        if len(lines) >= limit:
            break
    return lines
