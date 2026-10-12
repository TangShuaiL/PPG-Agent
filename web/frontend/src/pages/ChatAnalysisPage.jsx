import React, { useRef, useEffect, useState } from 'react'
import {
  Card, Space, Input, Button, Tag, Spin, message,
} from 'antd'
import { SendOutlined, UserOutlined } from '@ant-design/icons'
import { api } from '../api.js'
import {
  SegmentPicker, WaveformFrame, WaveformPlaceholder,
  UnifiedWaveformPanel, AnalysisDetails,
  SegmentWaveformTabs, AdaptControls, LivePredictPanel,
} from '../components/AnalysisViews.jsx'
import AgentTraceCard from '../components/AgentTraceCard.jsx'

function Bubble({ role, content }) {
  const isUser = role === 'user'
  return (
    <div style={{
      maxWidth: '80%', minWidth: 0, flex: '0 1 auto',
      background: isUser ? '#1677ff' : '#fff',
      color: isUser ? '#fff' : '#333',
      borderRadius: 8, padding: '8px 12px',
      boxShadow: '0 1px 2px rgba(0,0,0,.06)',
      whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'break-word',
      fontSize: 14, lineHeight: 1.6,
    }}>
      {content}
    </div>
  )
}

export default function ChatAnalysisPage() {
  const [pick, setPick] = useState({ source: 'vital', subjectId: null, segId: null })
  // shared analysis target: the fixed-axis canvas highlight layer and the
  // live panel both key off this one control (切目标只换高亮，轴/波形不动)
  const [target, setTarget] = useState('sbp')
  const [messages, setMessages] = useState([
    { role: 'assistant',
      content: '你好，我是 PPG-Agent 分析助手。先在上方选择一个数据集 / 受试者 / segment，然后告诉我你想做什么——例如"分析这个波形的信号质量"或"这个段和我自己的历史比有什么变化"。' },
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [analyzedSeg, setAnalyzedSeg] = useState(null)
  const [adapt, setAdapt] = useState(null)
  // 对话触发的 L3 可解释载荷：与右侧"生成解释"按钮共用同一渲染（下传给 LivePredictPanel）
  const [explain, setExplain] = useState(null)
  const [explainedSeg, setExplainedSeg] = useState(null)
  const listRef = useRef(null)

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, busy])

  const send = async (overrideText) => {
    // onClick/onPressEnter pass the DOM event — only treat strings as text
    const text = (typeof overrideText === 'string' ? overrideText : input).trim()
    if (!text) return
    // 右侧便捷按钮的语义是“替你把指令发给 Agent”，忙时要回显而不是静默丢弃
    if (busy) { message.info('Agent 正在处理上一条请求，请稍候…'); return }
    setInput('')
    setMessages((m) => [...m, { role: 'user', content: text }])
    setBusy(true)
    try {
      const history = messages.slice(-12).map(({ role, content }) => ({ role, content }))
      const r = await api.chatMessage({
        message: text, history,
        source: pick.source, subject_id: pick.subjectId, seg_id: pick.segId, target,
      })
      setMessages((m) => [...m, {
        role: 'assistant', content: r.reply, trace: r.agent_execution,
      }])
      if (r.analysis) {
        setAnalysis(r.analysis)
        setAnalyzedSeg(r.seg_id ?? pick.segId)
      }
      if (r.explain) {
        setExplain(r.explain)
        setExplainedSeg(r.seg_id ?? pick.segId)
      }
      if (r.adapt) setAdapt(r.adapt)
      if (r.adapt?.status !== 'active' && r.adapt && !r.analysis) setAnalysis(null)
    } catch (e) {
      message.error(`对话失败: ${e.message}`)
      setMessages((m) => [...m, { role: 'assistant', content: `⚠️ 请求失败：${e.message}` }])
    } finally {
      setBusy(false)
    }
  }

  // 对话产出的解释载荷：仅当对应段与目标匹配时下传（画布历史 Tab 与 L3 面板同源）
  const gatedExplain = (explainedSeg === pick.segId && explain && explain.target === target)
    ? explain : null

  return (
    <div style={{
      // 绝对定位锚定到 Content（App.jsx 中 position:relative）：尺寸直接
      // 取自包含块 padding-box，是确定值，不依赖 flex/百分比高度链的逐层
      // 解析；inset 16 对应 Content 的 padding
      position: 'absolute', top: 16, left: 16, right: 16, bottom: 16,
      display: 'flex', flexDirection: 'column', gap: 16, overflow: 'hidden',
    }}>
      <Card size="small">
        <Space wrap>
          <SegmentPicker value={pick} onChange={setPick} />
          <Tag color={pick.segId ? 'blue' : 'default'}>
            {pick.segId ? `已选 segment #${pick.segId}` : '未选择 segment'}
          </Tag>
        </Space>
      </Card>

      <div style={{ flex: 1, minHeight: 0, display: 'flex', gap: 16 }}>
        {/* overflow: hidden ⇒ min-height 归零（flex item 默认 min-height:auto
            会被消息内容撑高，导致页面级滚动），滚动只发生在消息列表内 */}
        <div style={{ flex: '0 0 37.5%', minWidth: 0, minHeight: 0, display: 'flex' }}>
          <Card size="small" title="对话"
                style={{ flex: 1, width: '100%', display: 'flex', flexDirection: 'column',
                         minHeight: 0, overflow: 'hidden' }}
                styles={{ body: { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' } }}>
            <div ref={listRef} style={{
              flex: 1, minHeight: 0, overflow: 'auto', paddingRight: 4,
              background: '#fafafa', borderRadius: 6, padding: 12, marginBottom: 12,
            }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: '100%' }}>
              {messages.map((m, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'flex-start', gap: 8,
                  maxWidth: '100%',
                  flexDirection: m.role === 'user' ? 'row-reverse' : 'row',
                }}>
                  <div style={{ marginTop: 4, flexShrink: 0 }}>
                    {m.role === 'user' ? (
                      <UserOutlined style={{ fontSize: 20, color: '#999' }} />
                    ) : (
                      <img src="/ppg-agent-logo.png" alt="PPG-Agent"
                           style={{ width: 26, height: 26, borderRadius: 5 }} />
                    )}
                  </div>
                  {m.role === 'assistant' ? (
                    <div style={{ minWidth: 0, maxWidth: '85%' }}>
                      <Bubble role={m.role} content={m.content} />
                      {m.trace?.length > 0 && <AgentTraceCard trace={m.trace} />}
                    </div>
                  ) : (
                    <Bubble role={m.role} content={m.content} />
                  )}
                </div>
              ))}
              {!busy && adapt?.status === 'active' && adapt.source === pick.source && adapt.pending_options?.length > 0 && (
                <div style={{
                  marginBottom: 12, padding: 10, background: '#fff',
                  border: '1px solid #d9d9d9', borderRadius: 8,
                }}>
                  <div style={{ fontSize: 12, color: '#888', marginBottom: 6 }}>
                    ⚡ Agent 提议的参数调整选项（点选后会自动发送）：
                  </div>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {adapt.pending_options.map((o, i) => (
                      <Button key={i} size="small" block
                              onClick={() => send(`选 ${'ABCDEF'[i]}`)}>
                        <Space>
                          <Tag color="blue">{'ABCDEF'[i]}</Tag>
                          <span style={{ fontSize: 13 }}>
                            {o.user_desc || `${o.param} → ${String(o.value)}`}
                          </span>
                        </Space>
                      </Button>
                    ))}
                    <Button size="small" block type="dashed"
                            onClick={() => send('可以了，保存吧')}>
                      我满意当前效果，不需要调整 —— 保存参数
                    </Button>
                  </Space>
                </div>
              )}
              {busy && (
                <div style={{ padding: '8px 12px' }}>
                  <Spin size="small" /> <span style={{ fontSize: 12, color: '#888' }}>Agent 思考中…</span>
                </div>
              )}
              </div>
            </div>
            {!busy && adapt?.status === 'active' && adapt.source === pick.source && (
              <Space wrap style={{ marginBottom: 8, width: '100%' }}>
                <Button type="primary" size="small"
                        onClick={() => send('可以了，保存吧')}>
                  保存参数
                </Button>
                <Button size="small" onClick={() => send('再看几个 segment')}>
                  换一批段
                </Button>
                <Button size="small" danger onClick={() => send('不调了，取消')}>
                  取消
                </Button>
              </Space>
            )}
            <Space.Compact style={{ width: '100%' }}>
              <Input
                value={input} placeholder="例如：分析这个波形的信号质量 / 解读特征的生理含义"
                onChange={(e) => setInput(e.target.value)}
                onPressEnter={send} disabled={busy}
              />
              <Button type="primary" icon={<SendOutlined />} onClick={send} loading={busy}>
                发送
              </Button>
            </Space.Compact>
          </Card>
        </div>
        <div style={{
          flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column',
          minHeight: 0, overflow: 'hidden',
        }}>
          {/* 波形画布固定置顶：下方文本/解释再长也只在内层滚动，不会把可视化挤走 */}
          <div style={{ flex: '0 0 auto' }}>
            <WaveformFrame>
              {analysis ? (
                <UnifiedWaveformPanel
                  source={pick.source} subjectId={pick.subjectId} segIdx={pick.segIdx}
                  segId={pick.segId} analysis={analyzedSeg === pick.segId ? analysis : null}
                  target={target} onTargetChange={setTarget}
                  explain={gatedExplain} onAskAgent={send}
                />
              ) : adapt && adapt.rounds?.[adapt.rounds.length - 1]?.samples?.length ? (
                <SegmentWaveformTabs samples={adapt.rounds[adapt.rounds.length - 1].samples} />
              ) : pick.segId ? (
                <UnifiedWaveformPanel
                  source={pick.source} subjectId={pick.subjectId} segIdx={pick.segIdx}
                  segId={pick.segId} analysis={null}
                  target={target} onTargetChange={setTarget}
                  explain={gatedExplain} onAskAgent={send}
                />
              ) : (
                <WaveformPlaceholder />
              )}
            </WaveformFrame>
          </div>
          <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <LivePredictPanel
              source={pick.source} subjectId={pick.subjectId} segIdx={pick.segIdx} target={target}
              explain={gatedExplain} onAskAgent={send}
            />
            {analysis && analyzedSeg === pick.segId && <AnalysisDetails analysis={analysis} />}
            {adapt && <AdaptControls adapt={adapt} />}
          </div>
        </div>
      </div>
    </div>
  )
}
