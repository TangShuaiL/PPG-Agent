// Agent execution trace — a collapsible card under each assistant message
// showing the supervisor plan → sub-agent stages → synthesis, mirroring the
// project's multi-agent orchestration on the frontend.
import React from 'react'
import { Collapse, Tag } from 'antd'
import { ApiOutlined } from '@ant-design/icons'

const AGENT_META = {
  supervisor: { label: '主管 Agent', color: 'purple' },
  signal_processing: { label: 'Agent 1 · 信号处理', color: 'blue' },
  cardiovascular: { label: 'Agent 2 · 表型分析', color: 'cyan' },
  ml_prediction: { label: 'Agent 3 · ML 预测', color: 'gold' },
}

const STAGE_META = {
  plan: '意图识别与任务分配',
  dispatch: '任务派发',
  synthesize: '汇总编排',
  fallback: '降级',
  graph_error: '图执行错误',
  // Agent 1 graph nodes
  load: '加载信号', normalize: '归一化', scan: '质量扫描',
  segment: 'beat 分割', extract_features: '特征提取', decide: '决策',
  report_error: '处理错误', tools_decide: '决策工具', tools_adapt: '适配工具',
  // Agent 2 graph nodes
  assess: '表型评估', interpret: 'LLM 解读', tools_assess: '评估工具',
  // adaptation session
  adapt: '适配会话', predict: '模型预测',
  // ReAct-fallback tool stages
  analyze_signal: '信号分析', analyze_phenotype: '表型分析',
  ml_predict_risk: 'ML 预测', start_dataset_adaptation: '启动适配',
  propose_adaptation_options: '提议选项', apply_adaptation_choice: '应用选项',
  sample_more_segments: '更换采样', save_dataset_adaptation: '保存参数',
  cancel_dataset_adaptation: '取消适配',
}

export default function AgentTraceCard({ trace }) {
  if (!trace?.length) return null
  const items = [{
    key: 'trace',
    label: (
      <span style={{ fontSize: 12, color: '#666' }}>
        <ApiOutlined style={{ marginRight: 6 }} />
        Agent 执行轨迹
        <span style={{ marginLeft: 8, color: '#aaa', fontWeight: 400 }}>
          主管 → 子 Agent → 汇总
        </span>
      </span>
    ),
    children: (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {trace.map((e, i) => {
          const am = AGENT_META[e.agent] || { label: e.agent, color: 'default' }
          const sm = STAGE_META[e.stage] || e.stage
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: 12 }}>
              <Tag color={am.color} style={{ width: 118, textAlign: 'center', margin: 0, flexShrink: 0 }}>
                {am.label}
              </Tag>
              <div style={{ minWidth: 0 }}>
                <span style={{ color: '#555' }}>{sm}</span>
                {e.summary && (
                  <div style={{ color: '#888', wordBreak: 'break-word' }}>{e.summary}</div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    ),
  }]
  return (
    <Collapse size="small" ghost items={items} defaultActiveKey={[]}
              style={{ marginTop: 6, maxWidth: 620 }} />
  )
}
