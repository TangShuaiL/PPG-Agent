// Shared analysis result views (used by ChatAnalysisPage)
import React, { useEffect, useRef, useState } from 'react'
import {
  Row, Col, Card, Select, Table, Tag, Alert, Statistic, Descriptions,
  Space, message, Collapse, Button, Tabs, Segmented, Spin,
} from 'antd'
import { api } from '../api.js'
import ReactECharts from 'echarts-for-react'
import WaveformChart from './WaveformChart.jsx'
import { LEVEL_COLORS, SOURCE_COLORS, TIER_COLORS } from './waveTheme'

// re-exported for callers that imported them from here historically
export { SOURCE_COLORS, LEVEL_COLORS }

// ------------------------------------------------------------------
// Fixed visualization window: the frame is always drawn (empty state
// shows an axis placeholder); waveforms render inside it once ready.
// ------------------------------------------------------------------
export function WaveformFrame({ children }) {
  return (
    <div style={{
      height: 480, background: '#fff', borderRadius: 8,
      border: '1px solid #e8e8e8', overflow: 'hidden', marginBottom: 16,
      position: 'relative',
    }}>
      {children}
    </div>
  )
}

export function WaveformPlaceholder({ text = '等待波形 — 在对话中发起分析或数据集适配' }) {
  return (
    <svg width="100%" height="100%" viewBox="0 0 600 300"
         style={{ position: 'absolute', inset: 0, background: '#fff' }}>
      <rect x="50" y="25" width="525" height="230" fill="#fafafa" stroke="#d9d9d9" />
      {[...Array(5)].map((_, i) => (
        <line key={`h${i}`} x1="50" y1={25 + i * 57.5} x2="575" y2={25 + i * 57.5} stroke="#eee" />
      ))}
      {[...Array(9)].map((_, i) => (
        <line key={`v${i}`} x1={50 + i * 65.6} y1="25" x2={50 + i * 65.6} y2="255" stroke="#eee" />
      ))}
      {[...Array(6)].map((_, i) => (
        <text key={`xt${i}`} x={50 + i * 100} y="275" textAnchor="middle" fontSize="9" fill="#ccc">
          {i * 6}s
        </text>
      ))}
      {[...Array(5)].map((_, i) => (
        <text key={`yt${i}`} x="42" y={29 + i * 57.5} textAnchor="end" fontSize="9" fill="#ccc">
          {i * 25}
        </text>
      ))}
      <text x="310" y="150" textAnchor="middle" fontSize="14" fill="#bbb">{text}</text>
      <text x="575" y="285" textAnchor="end" fontSize="10" fill="#ccc">Time (s)</text>
      <text x="14" y="150" textAnchor="middle" fontSize="10" fill="#ccc"
            transform="rotate(-90 14 150)">Amplitude</text>
    </svg>
  )
}

// ------------------------------------------------------------------
// Segment picker: source → subject → segment (cascading)
// 默认只列 test 集受试者（基线在 train 上构建，test 集才是真正的
// "未见人群"演示）；可切换到 train 或全部。
// ------------------------------------------------------------------
export function SegmentPicker({ value, onChange }) {
  const [subjects, setSubjects] = useState([])
  const [total, setTotal] = useState(0)
  const [detail, setDetail] = useState(null)
  const [split, setSplit] = useState('test')
  const timer = useRef(null)

  // Load candidate subjects for the picker. `q` enables server-side
  // subject_id search so large test sets (e.g. vital ~558) stay reachable
  // beyond the page_size cap; page_size is the backend max (500).
  const loadSubjects = (q) => {
    if (!value.source) return
    api.subjects({ source: value.source, split: split || undefined,
                   q: q || undefined, page: 1, page_size: 500 })
      .then((d) => { setSubjects(d.items); setTotal(d.total) })
      .catch((e) => message.error(e.message))
  }

  useEffect(() => { loadSubjects('') }, [value.source, split])

  useEffect(() => {
    if (!value.source || !value.subjectId) { setDetail(null); return }
    api.subject(value.source, value.subjectId)
      .then(setDetail)
      .catch((e) => message.error(e.message))
  }, [value.source, value.subjectId])

  return (
    <Space wrap>
      <Segmented
        value={split} size="small"
        onChange={(v) => { setSplit(v); onChange({ ...value, subjectId: null, segId: null, segIdx: null }) }}
        options={[
          { value: 'test', label: 'test' },
          { value: 'train', label: 'train' },
          { value: '', label: '全部' },
        ]}
      />
      <Select
        value={value.source} style={{ width: 120 }}
        onChange={(v) => onChange({ source: v, subjectId: null, segId: null, segIdx: null })}
        options={['vital', 'mimic', 'aurora'].map((s) => ({ value: s, label: s.toUpperCase() }))}
      />
      <Select
        value={value.subjectId} style={{ width: 200 }} showSearch
        filterOption={false} placeholder={`受试者 (共 ${total})`}
        onSearch={(q) => {
          clearTimeout(timer.current)
          timer.current = setTimeout(() => loadSubjects(q), 250)
        }}
        onChange={(v) => onChange({ ...value, subjectId: v, segId: null, segIdx: null })}
        options={subjects.map((s) => ({
          value: s.subject_id,
          label: `${s.subject_id}`,
        }))}
      />
      <Select
        value={value.segId} style={{ width: 200 }} placeholder="segment"
        onChange={(v) => {
          const seg = (detail?.segments || []).find((s) => s.id === v)
          onChange({ ...value, segId: v, segIdx: seg ? seg.seg_idx : null })
        }}
        options={(detail?.segments || []).map((s) => ({
          value: s.id,
          label: `#${s.seg_idx}`,
        }))}
      />
      {detail?.subject?.gender && <Tag>{detail.subject.gender}</Tag>}
    </Space>
  )
}

// ------------------------------------------------------------------
// L1 可解释性：Agent 3 段级实时预测 + 近窗历史注意力归因
//   当前段预测值 vs 实测；并展示本次估计"参考了哪些前序历史段"
//   （c_recent 注意力，仅覆盖最近 K 段）。群体统计估计，非诊断。
// ------------------------------------------------------------------
const ATT_TARGET_COLORS = { sbp: '#f5222d', dbp: '#fa8c16', age: '#1677ff' }

// Prediction numbers + L3 narrative for the live chain.  Feature-point /
// attention highlights live on the unified waveform canvas (see
// UnifiedWaveformPanel); `target` is lifted to the page so both stay in sync.
export function LivePredictPanel({ source, subjectId, segIdx, target, explain, onAskAgent }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState(null)
  const reqId = useRef(0)
  // L3 解释叙事：由 Agent 对话产出（explain prop）——不再右侧直连端点
  const [expl, setExpl] = useState(null)
  // 测试集真值里的年龄标签是受试者级（段表无 age），从 subject 详情取
  const [ageLabel, setAgeLabel] = useState(null)

  useEffect(() => {
    if (!source || !subjectId || segIdx == null) { setData(null); setErr(null); return }
    const myId = ++reqId.current
    setLoading(true); setErr(null)
    api.livePredict(source, subjectId, segIdx)
      .then((d) => { if (reqId.current === myId) setData(d) })
      .catch((e) => { if (reqId.current === myId) { setData(null); setErr(e.message) } })
      .finally(() => { if (reqId.current === myId) setLoading(false) })
  }, [source, subjectId, segIdx])

  // 年龄真值：subject 级（段表无 age），单独取一次
  useEffect(() => {
    if (!source || !subjectId) { setAgeLabel(null); return }
    let live = true
    api.subject(source, subjectId)
      .then((d) => { if (live) setAgeLabel(d?.subject?.age ?? null) })
      .catch(() => { if (live) setAgeLabel(null) })
    return () => { live = false }
  }, [source, subjectId])

  // 目标/段切换时失效旧叙事（归因高亮层已迁至统一画布）
  useEffect(() => {
    setExpl(null)
  }, [source, subjectId, segIdx, target])

  // L3 解释由 Agent 对话产出：explain prop 一到即填入（与统一画布历史 Tab 同源）。
  // 置于失效 effect 之后：切到不匹配目标时 explain prop 为 null，此 effect 不动作。
  useEffect(() => {
    if (explain) setExpl(explain)
  }, [explain])

  // “生成解释”= 替你把指令发给 Agent（路由到 explain_prediction 工具），不再直连端点
  const TARGET_CN = { sbp: '收缩压', dbp: '舒张压', age: '年龄' }
  const askExplain = () => onAskAgent?.(
    `请解释这个段当前的${TARGET_CN[target] || target}预测为什么是这个值，主要看哪些波形特征点`)

  if (!source || !subjectId || segIdx == null) return null

  const pred = data?.predictions || {}
  const meas = data?.measured_bp || null
  const degraded = data?.engine && data.engine !== 'live_stream'

  return (
    <Card size="small" style={{ marginTop: 16 }}
          title="实时推理可解释性 (Agent 3)"
          extra={<Space size={4}>
            {data && <Tag>{data.engine}</Tag>}
            {data && <Tag color={data.history_source === 'table' ? 'green' : 'orange'}>
              历史:{data.history_source}</Tag>}
            {data && <Tag>n_hist {data.n_history}</Tag>}
          </Space>}>
      {loading && <div style={{ padding: 12, color: '#888', fontSize: 12 }}>
        实时编码当前段并计算注意力中…（首次加载 GPU 编码器较慢）</div>}
      {err && <Alert type="error" showIcon message="推理失败" description={err} />}
      {!loading && !err && data && (
        <div>
          {degraded && data.live_unavailable_reason && (
            <Alert type="warning" showIcon style={{ marginBottom: 8 }}
                   message="实时链不可用，已降级为查表口径（同头同嵌入，逐位等价）"
                   description={data.live_unavailable_reason} />
          )}
          <Row gutter={16} style={{ marginBottom: 8 }}>
            {['sbp', 'dbp', 'age'].map((t) => (
              <Col span={8} key={t}>
                <Statistic title={`${t.toUpperCase()} 预测`} value={pred[t] ?? '—'}
                           valueStyle={{ fontSize: 20, color: ATT_TARGET_COLORS[t] }} />
              </Col>
            ))}
          </Row>
          <div style={{ fontSize: 11, color: '#aaa', marginTop: -4, marginBottom: 6 }}>
            推理为无标签场景：预测值不与实测对照展示（真值见下方“测试集真值”模块）。
          </div>
          <div style={{ border: '1px dashed #d9d9d9', borderRadius: 6, padding: '6px 8px', marginBottom: 6, background: '#fcfbf7' }}>
            <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>
              测试集真值标签（仅供评估，不参与模型推理）
            </div>
            <Row gutter={16}>
              {[['SBP', meas?.sbp, pred.sbp], ['DBP', meas?.dbp, pred.dbp],
                ['AGE', ageLabel, pred.age]].map(([name, truth, p]) => {
                const has = truth != null && !Number.isNaN(truth)
                return (
                  <Col span={8} key={name}>
                    <Statistic title={`${name} 真值`}
                               value={has ? Number(truth).toFixed(1) : '—'}
                               suffix={has && p != null
                                 ? <span style={{ fontSize: 12, color: '#999' }}>
                                     Δ{Math.abs(p - truth).toFixed(1)}
                                   </span> : null}
                               valueStyle={{ fontSize: 16, color: has ? '#8c8c8c' : '#bbb' }} />
                  </Col>
                )
              })}
            </Row>
            {!meas?.sbp && !meas?.dbp && (
              <div style={{ fontSize: 11, color: '#aaa' }}>该段无可用 BP 标签（无标签或已过离谱闸门剔除）。</div>)}
          </div>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>
            当前分析目标：<b style={{ color: ATT_TARGET_COLORS[target] }}>{target.toUpperCase()}</b>
            {' '}· 特征点归因与历史段注意力已高亮在上方波形画布上（切目标/切段只改高亮层）。
          </div>
          <div style={{ marginTop: 10, paddingTop: 8, borderTop: '1px solid #f0f0f0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, color: '#666' }}>L3 解释叙事—为什么出这个值（{target}）</span>
              <Button size="small" type="primary" ghost onClick={askExplain}>
                {expl ? '重新解释' : '让 Agent 解释'}</Button>
            </div>
            {expl?.unavailable && (
              <Alert type="warning" showIcon message="解释链不可用（需 GPU+encoder）"
                     description={expl.unavailable} />)}
            {expl && !expl.unavailable && (
              <div style={{ background: '#fafafa', borderRadius: 6, padding: '8px 10px' }}>
                <div style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 6 }}>{expl.direct_answer}</div>
                {(expl.current_segment?.top_keypoints || []).length > 0 && (
                  <div style={{ marginBottom: 6 }}>
                    <span style={{ fontSize: 12, color: '#666' }}>当前段特征点：</span>
                    <Space size={6} wrap style={{ marginLeft: 4 }}>
                      {expl.current_segment.top_keypoints.map((f) => (
                        <Tag key={f.name} color={f.direction === 'up' ? 'red' : 'blue'}>
                          {f.label} {f.contrib >= 0 ? '+' : ''}{f.contrib}
                        </Tag>
                      ))}
                    </Space>
                  </div>
                )}
                {(expl.history?.segments || []).length > 0 && (
                  <div style={{ marginBottom: 6 }}>
                    <span style={{ fontSize: 12, color: '#666' }}>历史段特征点：</span>
                    <div style={{ marginTop: 2 }}>
                      {expl.history.segments.map((s) => (
                        <div key={s.seg_idx} style={{ fontSize: 12, marginTop: 2 }}>
                          <b>#{s.seg_idx}</b>
                          <span style={{ color: '#aaa' }}> 注意力 {s.attention_weight} · 净贡献 {s.net_contrib}</span>
                          {' '}· {(s.top_keypoints || []).map((f) => f.label).join('、')}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {(expl.knowledge_interpretation?.statements || []).length > 0 && (
                  <ul style={{ paddingLeft: 18, margin: '4px 0', fontSize: 12, color: '#555', lineHeight: 1.6 }}>
                    {expl.knowledge_interpretation.statements.map((s, i) => (<li key={i}>{s}</li>))}
                  </ul>
                )}
                {(expl.uncertainty || []).length > 0 && (
                  <div style={{ fontSize: 11, color: '#d48806', marginTop: 4 }}>
                    不确定性：{expl.uncertainty.join(' ')}
                  </div>
                )}
                <div style={{ fontSize: 11, color: '#999', marginTop: 6 }}>{expl.safety_note}</div>
              </div>
            )}
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: '#999', lineHeight: 1.5 }}>
            {data.history_attention_note || '近窗注意力为群体模型依赖度证据，'
              + '非个体生理病因；模型输出是群体统计估计，不作诊断。'}
          </div>
        </div>
      )}
    </Card>
  )
}

// ------------------------------------------------------------------
// 跨人群百分位对比（描述性统计，保留）——证伪的 VAI 仪表盘已随叙事层
// 清理删除（compute_vai 仅保留在引擎层供 analysis/ 复现脚本）
// ------------------------------------------------------------------
export function CrossPopulationPanel({ crossPop }) {
  const rows = Object.entries(crossPop || {}).flatMap(([feature, r]) =>
    Object.entries(r.results || {}).map(([src, q]) => ({ key: `${feature}-${src}`, feature, source: src, ...q })))
  if (!rows.length) return null
  return (
    <div>
      <Table
        size="small" pagination={false} rowKey="key" dataSource={rows}
        columns={[
          { title: '特征', dataIndex: 'feature' },
          { title: '基线', dataIndex: 'source',
            render: (s) => <Tag color={SOURCE_COLORS[s]}>{s}</Tag> },
          { title: '百分位', dataIndex: 'percentile' },
          { title: '水平', dataIndex: 'level',
            render: (l) => (l ? <Tag color={LEVEL_COLORS[l]}>{l}</Tag> : '—') },
          { title: 'P50', dataIndex: 'p50', render: (v) => v?.toFixed?.(3) },
        ]}
      />
      {rows[0] && crossPop[rows[0].feature]?.interpretation && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
          {Object.values(crossPop)[0].interpretation}
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------
// Single-axis waveform viewer: fixed-height window + horizontal tabs
// to switch between sample segments.  Used by the dataset adaptation
// panel; designed as the shared pattern for all single-axis-waveform
// visualizations (waveform + highlighted regions of interest).
// ------------------------------------------------------------------
export function SegmentWaveformTabs({ samples }) {
  if (!samples?.length) return null
  const items = samples.map((s, i) => {
    const m = s.metrics || {}
    const tier = m.tier || m.quality_tier
    return {
      key: String(i),
      label: (
        <Space size={4}>
          <span>{s.name}</span>
          {m.error
            ? <Tag color="red">error</Tag>
            : <Tag color={TIER_COLORS[tier]}>{Number(m.quality).toFixed(2)}</Tag>}
        </Space>
      ),
      children: m.error ? (
        <Alert type="error" showIcon message={m.error} />
      ) : (
        <SampleWaveform sample={s} />
      ),
    }
  })
  return <Tabs size="small" items={items} />
}

// One adapted sample: metrics + unified ECharts waveform (beat segmentation
// lines + fiducial keypoints + 特征点名 toggle) + per-beat feature curves.
// Falls back to the legacy PNG for sessions without a waveform payload.
function SampleWaveform({ sample }) {
  const [labels, setLabels] = useState('first') // 特征点名: first → all → none
  const m = sample.metrics || {}
  const wf = sample.waveform
  const labelText = { first: '首beat', all: '全部', none: '隐藏' }[labels]
  const cycle = () =>
    setLabels((v) => (v === 'first' ? 'all' : v === 'all' ? 'none' : 'first'))
  return (
    <>
      <div style={{ marginBottom: 8 }}><MetricsRow metrics={m} /></div>
      {wf ? (
        <>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 4 }}>
            <Button size="small" onClick={cycle}>特征点名: {labelText}</Button>
          </div>
          <WaveformChart
            points={wf.points} fs={wf.fs} height={300}
            durationSec={wf.duration_sec} overlay={wf.overlay}
            showKeypoints showBoundaries keypointLabels={labels}
          />
          <FeatureCurvesChart beats={wf.overlay?.beats} />
        </>
      ) : (
        // No matplotlib PNG fallback: an adapt sample without a waveform
        // payload shows the same loading state as the analysis canvas, so a
        // stage never renders in two visual styles.
        <div style={{ padding: 24, textAlign: 'center', color: '#888', fontSize: 12 }}>
          <Spin size="small" /> 该段波形加载中…
        </div>
      )}
    </>
  )
}

// Per-beat feature curves (adapt mode): x = beat index, one feature at a time,
// drawn from the same overlay feats as the waveform keypoints.
const CURVE_FEATURES = ['reflection_index', 'lasi', 'slope_md',
                        'pulse_width_50', 't_sys_ms', 'beat_len_ms',
                        'perfusion_index']

export function FeatureCurvesChart({ beats, height = 220 }) {
  const [feature, setFeature] = useState(null)
  const available = []
  const seen = new Set()
  for (const b of beats || []) {
    for (const f of Object.keys(b.feats || {})) {
      if (!seen.has(f)) { seen.add(f); available.push(f) }
    }
  }
  const picked = feature && available.includes(feature)
    ? feature
    : (CURVE_FEATURES.find((f) => available.includes(f)) ?? available[0])
  if (!picked) return null
  const curve = (beats || [])
    .filter((b) => b.feats?.[picked] != null)
    .map((b) => ({ x: b.beat_idx, y: +Number(b.feats[picked]).toFixed(4) }))
    .filter((p) => Number.isFinite(p.y))
  const option = {
    animation: false,
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    tooltip: { trigger: 'item',
               formatter: (p) => `B${p.data[0]} · ${picked}: ${p.data[1]}` },
    xAxis: { type: 'value', min: -0.5, max: (beats?.length || 1) - 0.5,
             name: 'beat', nameLocation: 'middle', nameGap: 22 },
    yAxis: { type: 'value', scale: true },
    series: [
      { type: 'line', showSymbol: false, lineStyle: { width: 1, color: '#1677ff' },
        data: curve.map((p) => [p.x, p.y]) },
      { type: 'scatter', symbolSize: 5, itemStyle: { color: '#1677ff' },
        data: curve.map((p) => [p.x, p.y]) },
    ],
  }
  return (
    <div style={{ marginTop: 8, padding: '4px 6px', borderTop: '1px solid #f0f0f0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: '#666' }}>逐 beat 特征曲线</span>
        <Select size="small" style={{ width: 200 }} value={picked}
                onChange={setFeature}
                options={available.map((f) => ({ value: f, label: f }))} />
      </div>
      <ReactECharts option={option} style={{ height }} notMerge lazyUpdate />
    </div>
  )
}

// ------------------------------------------------------------------
// Dataset adaptation (chat flow): sample segments + metrics + params
// ------------------------------------------------------------------
export function MetricsRow({ metrics }) {
  if (!metrics) return null
  const tier = metrics.tier || metrics.quality_tier
  return (
    <Space wrap style={{ fontSize: 13 }}>
      <Tag color={TIER_COLORS[tier]}>
        质量 {metrics.quality} ({tier})
      </Tag>
      <span>beats {metrics.n_valid}/{metrics.n_total}</span>
      <span>coverage {metrics.coverage}</span>
      <span>valid_ratio {metrics.valid_ratio}</span>
      {metrics.n_rejected_mad > 0 && (
        <Tag color="red">MAD剔除 {metrics.n_rejected_mad}</Tag>
      )}
      <span>amp CV {metrics.cv_pulse_amp}</span>
      <span>t_sys CV {metrics.cv_t_sys_ms}</span>
      <span>beat_len CV {metrics.cv_beat_len_ms}</span>
    </Space>
  )
}

const OPTION_LETTERS = 'ABCDEF'

// Adaptation info panel (right side): status, change history, current params.
// All interaction lives in the chat panel (option buttons + quick actions),
// so this stays read-only to avoid duplicated controls.
export function AdaptControls({ adapt }) {
  const changes = (adapt.rounds || []).filter((r) => r.change).map((r) => r.change)
  const active = adapt.status === 'active'
  return (
    <Card size="small" title={`数据集适配 — ${adapt.source}`}
          extra={active
            ? <Tag color="blue">进行中</Tag>
            : <Tag color={adapt.status === 'saved' ? 'green' : 'default'}>{adapt.status}</Tag>}
          style={{ marginBottom: 16 }}>
      {adapt.existing_presets?.length > 0 && (
        <Alert type="info" showIcon style={{ marginBottom: 12 }}
               message={`该数据集已有适配参数: ${adapt.existing_presets.join(', ')}`}
               description="本次适配不会覆盖它们——满意后保存为 user_<数据集> 预设，可随时清理。" />
      )}
      {changes.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          {changes.map((c, i) => (
            <Alert key={i} type="success" showIcon style={{ marginBottom: 4 }}
                   message={`已调整 ${c.param}: ${c.old_value} → ${c.new_value}`
                            + (c.rationale ? ` — ${c.rationale}` : '')} />
          ))}
        </div>
      )}
      <Collapse size="small" items={[{
        key: 'params',
        label: '当前提取参数',
        children: (
          <Descriptions size="small" column={3} bordered>
            {Object.entries(adapt.params || {}).map(([k, v]) => (
              <Descriptions.Item key={k} label={k}>{String(v)}</Descriptions.Item>
            ))}
          </Descriptions>
        ),
      }]} />
    </Card>
  )
}

// ------------------------------------------------------------------
// Analysis waveform — renders inside the fixed WaveformFrame
// ------------------------------------------------------------------
export function AnalysisWaveform({ segId, analysis }) {
  const [waveform, setWaveform] = useState(null)
  const [showKeypoints, setShowKeypoints] = useState(true)
  const [highlightFeature, setHighlightFeature] = useState(null)

  useEffect(() => {
    if (!segId) return
    setWaveform(null)
    setHighlightFeature(null)
    api.waveform(segId, 5000).then(setWaveform).catch(() => {})
  }, [segId])

  // Abnormal features with per-beat annotation (enriched server-side into
  // overlay.beats[].feats), ordered by the report's stratified_percentiles.
  const present = new Set()
  for (const b of (analysis?.overlay?.beats || [])) {
    for (const f of Object.keys(b.feats || {})) present.add(f)
  }
  const report = analysis?.report
  const abnormal = (Object.keys(report?.stratified_percentiles || {})
    .filter((f) => present.has(f)))
  const featOptions = abnormal.map((f) => {
    const level = report.stratified_percentiles[f].level
    return {
      value: f,
      label: (
        <Space size={6}>
          {f}
          <Tag color={LEVEL_COLORS[level]} style={{ marginInlineEnd: 0 }}>{level}</Tag>
        </Space>
      ),
    }
  })

  const beats = analysis?.beats
  if (!segId) return <WaveformPlaceholder text="等待波形 — 在对话中发起分析" />
  return (
    <div style={{ height: '100%', padding: 8, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        {featOptions.length > 0 && (
          <Select
            size="small" style={{ width: 240 }} allowClear
            placeholder="异常特征高亮 (逐 beat)"
            value={highlightFeature}
            onChange={setHighlightFeature}
            options={featOptions}
          />
        )}
        <div style={{ flex: 1 }} />
        <Button size="small" onClick={() => setShowKeypoints((v) => !v)}>
          {showKeypoints ? '隐藏关键点' : '显示关键点'}
        </Button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        {waveform ? (
          <WaveformChart
            points={waveform.points} fs={waveform.fs} height={430}
            durationSec={waveform.duration_sec}
            overlay={analysis?.overlay}
            showKeypoints={showKeypoints} keypointLabels="none"
            showBoundaries showValidity
            highlightFeature={highlightFeature}
          />
        ) : <div style={{ textAlign: 'center', padding: 24 }}>加载波形中…</div>}
      </div>
      {beats && (
        <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
          beats {beats.n_valid}/{beats.n_beats} 有效 ·
          质量 {beats.quality} ({beats.quality_tier}) · preset={beats.preset}
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------
// UnifiedWaveformPanel — the chat page's ONE always-mounted canvas on a
// FIXED axis (x = the 10 s normalized window, y = [-0.05, 1.05]).  The
// horizontal Tabs (当前 · Top-3 历史段) swap the trace *layer* only; the
// target Segmented swaps the *attribution highlight* only.  Neither
// re-inits the axes, so切 sbp/dbp/age 或跨段时波形零位移.
//   - current tab: points = api.waveform(segId), overlay = analysis.overlay,
//     highlight = api.liveAttrib(target).beat_points (per-beat IG), 逐拍异常 = report.
//   - history tab: only appears after the Agent's explain_prediction returns
//     (explain.history.segments); the first
//     time a tab opens we lazy-fetch api.segmentWaveform and cache it, so
//     toggling back never re-requests.  highlight = that segment's
//     feature_points; tab label carries its attention weight. Highlight uses
//     the segment's beat_points so IG lands on the specific beat/keypoint.
// All segments are min-max normalized to [0,1] over exactly 10 s in the
// library, so one fixed axis fits them and overlay keypoints (absolute
// 500 Hz samples) never drift from the downsampled display points.
// ------------------------------------------------------------------
export function UnifiedWaveformPanel({
  source, subjectId, segIdx, segId, analysis, target, onTargetChange,
  explain, onAskAgent,
}) {
  const [waveform, setWaveform] = useState(null)
  const [showKeypoints, setShowKeypoints] = useState(true)
  const [highlightFeature, setHighlightFeature] = useState(null)
  // current-segment IG attribution (需 GPU)
  const [attrib, setAttrib] = useState(null)
  const [aLoading, setALoading] = useState(false)
  const [aErr, setAErr] = useState(null)
  // Top-3 history attribution ← 由 Agent 解释载荷（explain.history.segments）派生，不再直连端点
  const [hAttrib, setHAttrib] = useState(null)
  // active tab + lazy-fetched/cached history payloads (key = String(seg_idx))
  const [activeTab, setActiveTab] = useState('current')
  const [histCache, setHistCache] = useState({})
  const [histLoading, setHistLoading] = useState({})
  const [histErr, setHistErr] = useState({})
  const reqId = useRef(0)

  // (re)analysis → reload current waveform, collapse tabs back to 当前, drop caches.
  useEffect(() => {
    const myId = ++reqId.current
    setWaveform(null); setHighlightFeature(null)
    setActiveTab('current'); setHistCache({}); setHistLoading({}); setHistErr({})
    if (!segId) return
    api.waveform(segId, 5000)
      .then((d) => { if (reqId.current === myId) setWaveform(d) })
      .catch(() => {})
  }, [segId])

  // attribution is target/segment-dependent → invalidate on change.
  useEffect(() => {
    setAttrib(null); setAErr(null)
    setHAttrib(null)
  }, [source, subjectId, segIdx, target])

  const fetchAttrib = () => {
    if (!source || !subjectId || segIdx == null) return
    setALoading(true); setAErr(null)
    api.liveAttrib(source, subjectId, segIdx, target)
      .then(setAttrib)
      .catch((e) => { setAttrib(null); setAErr(e.message) })
      .finally(() => setALoading(false))
  }

  // L2.5 历史段归因由 Agent 解释产出：explain.history.segments → 历史 Tabs（同源）
  useEffect(() => {
    const segs = (explain && !explain.unavailable)
      ? (explain.history?.segments || []) : []
    setHAttrib(segs.length ? { segments: segs } : null)
  }, [explain])

  // open a tab; lazy-fetch + cache a history segment's waveform on first visit.
  const openTab = (key) => {
    setActiveTab(key)
    if (key === 'current' || histCache[key] || histLoading[key]) return
    setHistLoading((s) => ({ ...s, [key]: true }))
    setHistErr((s) => ({ ...s, [key]: null }))
    api.segmentWaveform(source, subjectId, Number(key))
      .then((d) => setHistCache((c) => ({ ...c, [key]: d })))
      .catch((e) => setHistErr((s) => ({ ...s, [key]: e.message })))
      .finally(() => setHistLoading((s) => ({ ...s, [key]: false })))
  }

  // 异常特征高亮选项（当前段：report 分层百分位 ∩ overlay.beats[].feats）
  const present = new Set()
  for (const b of (analysis?.overlay?.beats || [])) {
    for (const f of Object.keys(b.feats || {})) present.add(f)
  }
  const report = analysis?.report
  const abnormal = Object.keys(report?.stratified_percentiles || {}).filter((f) => present.has(f))
  const featOptions = abnormal.map((f) => {
    const level = report.stratified_percentiles[f].level
    return {
      value: f,
      label: (
        <Space size={6}>
          {f}
          <Tag color={LEVEL_COLORS[level]} style={{ marginInlineEnd: 0 }}>{level}</Tag>
        </Space>
      ),
    }
  })

  const histSegments = hAttrib?.segments || []
  const isCurrent = activeTab === 'current'
  const activeHistSeg = isCurrent
    ? null
    : histSegments.find((s) => String(s.seg_idx) === activeTab)

  // active view payload (drives the single, always-mounted WaveformChart)
  const view = isCurrent
    ? {
        points: waveform?.points, fs: waveform?.fs,
        durationSec: waveform?.duration_sec, overlay: analysis?.overlay,
        attribution: attrib?.beat_points || null,
      }
    : {
        points: histCache[activeTab]?.points, fs: histCache[activeTab]?.fs,
        durationSec: histCache[activeTab]?.duration_sec,
        overlay: histCache[activeTab]?.overlay,
        attribution: activeHistSeg?.beat_points || null,
      }

  const tabItems = [
    { key: 'current', label: '当前段' },
    ...histSegments.map((s, i) => ({
      key: String(s.seg_idx),
      label: (
        <Space size={4}>
          <span>{`H${i + 1} · #${s.seg_idx}`}</span>
          <Tag color="purple" style={{ marginInlineEnd: 0 }}>注意力 {s.attention_weight}</Tag>
        </Space>
      ),
    })),
  ]

  const loadingCurrent = isCurrent && !waveform
  const loadingHist = !isCurrent && !histCache[activeTab] && !histErr[activeTab]
  const beats = analysis?.beats

  return (
    <div style={{ height: '100%', padding: 8, display: 'flex', flexDirection: 'column' }}>
      {/* controls: target (shared) + current-segment layers */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
        <Segmented size="small" value={target} onChange={onTargetChange}
                   options={['sbp', 'dbp', 'age']} />
        {isCurrent && featOptions.length > 0 && (
          <Select
            size="small" style={{ width: 220 }} allowClear
            placeholder="异常特征高亮 (逐 beat)"
            value={highlightFeature}
            onChange={setHighlightFeature}
            options={featOptions}
          />
        )}
        <Button size="small" loading={aLoading} onClick={fetchAttrib}>
          {attrib ? '重算当前段敏感度' : '当前段特征点高亮'}</Button>
        <Button size="small" onClick={() => onAskAgent?.(
            '展开这个段 Top-3 历史段的特征点归因')}>
          {hAttrib ? '重新展开 Top-3 历史段' : '展开 Top-3 历史段 (问 Agent)'}</Button>
        <div style={{ flex: 1 }} />
        <Button size="small" onClick={() => setShowKeypoints((v) => !v)}>
          {showKeypoints ? '隐藏关键点' : '显示关键点'}
        </Button>
      </div>
      {aErr && <div style={{ fontSize: 12, color: '#d4380d', marginBottom: 4 }}>当前段归因失败（需 GPU）：{aErr}</div>}

      {/* horizontal tab bar — switcher only; the chart below is never re-mounted */}
      <Tabs size="small" activeKey={activeTab} onChange={openTab} items={tabItems}
            style={{ marginBottom: 0 }} />

      {!isCurrent && (
        <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>
          历史段 #{activeHistSeg?.seg_idx} · 注意力 {activeHistSeg?.attention_weight}
          {activeHistSeg?.ig_total != null && (
            <> · 净贡献 {activeHistSeg.ig_total >= 0 ? '+' : ''}{activeHistSeg.ig_total}</>) }
          {activeHistSeg?.n_beats_used != null && <> · {activeHistSeg.n_beats_used} 拍</>}
        </div>
      )}

      <div style={{ flex: 1, minHeight: 0 }}>
        {loadingCurrent || loadingHist ? (
          <div style={{ textAlign: 'center', padding: 24, color: '#888' }}>
            <Spin size="small" /> 波形加载中…
          </div>
        ) : !isCurrent && histErr[activeTab] ? (
          <div style={{ textAlign: 'center', padding: 24, color: '#d4380d', fontSize: 12 }}>
            该历史段波形获取失败：{histErr[activeTab]}
          </div>
        ) : view.points ? (
          <WaveformChart
            points={view.points} fs={view.fs} height={330}
            durationSec={view.durationSec ?? 10}
            overlay={view.overlay}
            showKeypoints={showKeypoints} keypointLabels="none"
            showBoundaries showValidity
            highlightFeature={isCurrent ? highlightFeature : null}
            attribution={view.attribution}
            yMin={-0.05} yMax={1.05}
          />
        ) : <div style={{ textAlign: 'center', padding: 24 }}>无波形数据</div>}
      </div>

      {isCurrent && beats && (
        <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
          beats {beats.n_valid}/{beats.n_beats} 有效 ·
          质量 {beats.quality} ({beats.quality_tier}) · preset={beats.preset}
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------
// 描述性证据面板（引擎确定性输出，保留）：尺度警告、MDC 变化参考。
// 已证伪面板（VAI/表型亚型/疾病关联/窗口上下文/规则依据）随叙事层
// 清理删除，数据仍在传输层 report 中供研究追溯。
// All rendered as collapsible panels inside AnalysisDetails.
// ------------------------------------------------------------------

// Generic table over an unknown-shape record list: preferred columns
// first, remaining scalar fields appended, numerics auto-formatted.
function AutoTable({ rows, preferred = [] }) {
  if (!rows?.length) return null
  const keys = [
    ...preferred.filter((k) => rows[0][k] !== undefined),
    ...Object.keys(rows[0]).filter((k) => !preferred.includes(k) && !['key'].includes(k)),
  ]
  return (
    <Table size="small" rowKey={(_, i) => i} pagination={false} dataSource={rows}
           columns={keys.map((k) => ({
             title: k,
             dataIndex: k,
             render: (v) => (typeof v === 'number'
               ? (v === 0 ? '0' : (Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(3)))
               : (v ?? '—')),
           }))} />
  )
}

function ScaleWarningsPanel({ sw }) {
  return (
    <Alert type="warning" showIcon
           message={`尺度警告:${sw.features?.length ?? 0} 个特征`}
           description={sw.rule} />
  )
}

function ChangeReferencePanel({ cr }) {
  if (!cr.available) {
    return <Alert type="info" showIcon message="变化参考 (MDC) 不可用" description={cr.reason} />
  }
  const rows = Object.entries(cr.features || {}).map(([feature, v]) => ({ key: feature, feature, ...v }))
  return <AutoTable rows={rows} preferred={['feature', 'verdict', 'mdc', 'icc']} />
}

// ------------------------------------------------------------------
// Analysis details (跨人群对比、尺度警告、变化参考 MDC、Safety、
// 段级特征表) — one collapsible accordion below the waveform frame
// ------------------------------------------------------------------
export function AnalysisDetails({ analysis }) {
  const report = analysis?.report
  const feats = analysis?.features?.feature_means
  const items = []

  if (report) {
    if (report.scale_warnings?.available) {
      items.push({
        key: 'scale',
        label: `尺度警告 (${report.scale_warnings.features?.length ?? 0})`,
        children: <ScaleWarningsPanel sw={report.scale_warnings} />,
      })
    }
    if (report.change_reference) {
      items.push({ key: 'change', label: '变化参考 (MDC)', children: <ChangeReferencePanel cr={report.change_reference} /> })
    }
    if (report.cross_population && Object.keys(report.cross_population).length) {
      items.push({ key: 'cross', label: '跨人群对比', children: <CrossPopulationPanel crossPop={report.cross_population} /> })
    }
    if (report.safety_note) {
      items.push({
        key: 'safety',
        label: 'Safety Note',
        children: <Alert type="warning" showIcon message="Safety Note" description={report.safety_note} />,
      })
    }
  }

  if (feats && Object.keys(feats).length) {
    items.push({
      key: 'feats',
      label: `段级特征 (${Object.keys(feats).length} 项)`,
      children: (
        <Descriptions size="small" column={3} bordered>
          {Object.entries(feats).map(([k, v]) => (
            <Descriptions.Item key={k} label={k}>{v}</Descriptions.Item>
          ))}
        </Descriptions>
      ),
    })
  }

  if (!items.length) return null
  return <Collapse size="small" items={items} defaultActiveKey={['cross']} />
}
