import React from 'react'
import ReactECharts from 'echarts-for-react'
import {
  BG_VALID, BG_REJECTED, BG_ABNORMAL, BOUNDARY_COLOR,
  KP_COLORS, KP_LABELS,
} from './waveTheme'

// PPG waveform line chart — the ONE renderer for all three modes
// (quality / phenotype / adapt), with mode-driven annotation layers keyed
// off the overlay beats:
//   - showValidity:     valid/invalid beat background tints (quality mode)
//   - showBoundaries:   dashed beat-segmentation lines (adapt mode)
//   - showKeypoints / keypointLabels: fiducial markers + 特征点名
//   - highlightFeature: amber tint over beats whose per-beat level for the
//                       selected feature is abnormal + per-beat tooltip
//
// Time axis: `points` may be downsampled server-side while the overlay
// keypoints stay in absolute 500 Hz samples.  durationSec (the signal's real
// length) drives the x grid, so annotations never drift from the waveform —
// `durationSec ?? (points.length-1)/fs` keeps old callers unchanged.
export default function WaveformChart({
  points, fs, height = 260, overlay = null,
  showKeypoints = false, keypointLabels = 'none',
  showBoundaries = true, showValidity = true,
  highlightFeature = null, durationSec = null,
  yMin = null, yMax = null, attribution = null,
}) {
  const duration = durationSec ?? (points.length - 1) / fs
  const dt = points.length > 1 ? duration / (points.length - 1) : 0
  const timeOf = (sample) => sample / (overlay?.fs || fs)
  // amplitude at a wall-clock time (maps onto the downsampled point grid)
  const ampAt = (t) => {
    const i = Math.min(points.length - 1, Math.max(0, Math.round(t / dt)))
    return points[i]
  }

  const beats = overlay?.beats || []

  const lineSeries = {
    type: 'line',
    showSymbol: false,
    lineStyle: { width: 1, color: '#1677ff' },
    data: points.map((y, i) => [+(i * dt).toFixed(4), y]),
  }
  const series = [lineSeries]

  let hlCount = 0 // beats highlighted for the legend

  // ── Layer: validity tints (silent — never intercept tooltips)
  if (showValidity && beats.length) {
    series.push({
      type: 'line', data: [], silent: true, z: 1,
      markArea: {
        silent: true,
        data: beats.map((b) => [
          { xAxis: +timeOf(b.start).toFixed(4),
            itemStyle: { color: b.valid ? BG_VALID : BG_REJECTED, opacity: 0.55 } },
          { xAxis: Math.min(duration, +timeOf(b.end ?? b.start).toFixed(4)) },
        ]),
      },
    })
  }

  // ── Layer: abnormal-feature highlight (phenotype mode)
  if (highlightFeature && beats.length) {
    const hlData = beats
      .filter((b) => b.valid && b.feats?.[highlightFeature])
      .map((b) => {
        const f = b.feats[highlightFeature]
        hlCount += 1
        return [
          { xAxis: +timeOf(b.start).toFixed(4),
            name: `B${b.beat_idx}`,
            itemStyle: { color: BG_ABNORMAL, opacity: 0.6 },
            tooltip: {
              formatter: () =>
                `B${b.beat_idx} · ${highlightFeature}: ${f.v} · ${f.percentile} · ${f.level}`,
            } },
          { xAxis: Math.min(duration, +timeOf(b.end ?? b.start).toFixed(4)) },
        ]
      })
    if (hlData.length) {
      series.push({
        type: 'line', data: [], z: 1.5,
        markArea: { silent: false, data: hlData },
      })
    }
  }

  // ── Layer: beat segmentation lines
  if (showBoundaries && beats.length) {
    const times = new Set()
    for (const b of beats) {
      for (const v of [b.start, b.end]) {
        if (v == null) continue
        const t = +timeOf(v).toFixed(4)
        if (t >= 0 && t <= duration) times.add(t)
      }
    }
    series.push({
      type: 'line', data: [], silent: true, z: 2,
      markLine: {
        symbol: 'none', silent: true,
        lineStyle: { type: 'dashed', color: BOUNDARY_COLOR, width: 0.8, opacity: 0.7 },
        label: { show: false },
        data: [...times].map((t) => ({ xAxis: t })),
      },
    })
  }

  // ── Layer: fiducial keypoints (+ 特征点名)
  if (showKeypoints && beats.length) {
    const firstValid = beats.find((b) => b.valid && b.keypoints)
    const scatterData = []
    for (const beat of beats) {
      if (!beat.valid || !beat.keypoints) continue
      const showLabels =
        keypointLabels === 'all' || (keypointLabels === 'first' && beat === firstValid)
      for (const [name, sample] of Object.entries(beat.keypoints)) {
        const t = +timeOf(sample).toFixed(4)
        if (t > duration + 0.01) continue
        scatterData.push({
          value: [t, ampAt(t)],
          name,
          itemStyle: { color: KP_COLORS[name] || '#333' },
          label: showLabels
            ? { show: true, position: 'top', fontSize: 9, color: '#555',
                formatter: () => KP_LABELS[name] || name }
            : undefined,
        })
      }
    }
    series.push({
      type: 'scatter', symbolSize: 6, data: scatterData, z: 3,
      tooltip: { formatter: (p) => `${p.data.name} @ t=${p.data.value[0]} s` },
    })
  }

  // ── Layer: attribution highlight (IG feature-point contributions).
  // Preferred (localized) mode — `attribution` is a list of per-(beat,
  // keypoint) items {beat_idx, name, label, contrib, t(sec)} straight from
  // the engine's per-beat IG: each marker sits on the EXACT keypoint of the
  // EXACT beat that IG actually pointed at (NOT smeared across every beat).
  // color = sign (红=拉高 / 蓝=拉低), size ∝ |contrib|; negligible |contrib|
  // dropped so only the salient beats/keypoints stand out. Same fixed axis.
  let attribCount = 0
  if (attribution && attribution.length && attribution[0] && attribution[0].t != null) {
    const maxAbs = Math.max(1e-9, ...attribution.map((f) => Math.abs(f.contrib || 0)))
    const cut = 0.12 * maxAbs
    const data = []
    for (const f of attribution) {
      const c = f.contrib || 0
      if (Math.abs(c) < cut) continue
      const t = +(+f.t).toFixed(4)
      if (t < 0 || t > duration + 0.01) continue
      attribCount += 1
      data.push({
        value: [t, ampAt(t)], name: f.name,
        symbolSize: 5 + 15 * (Math.abs(c) / maxAbs),
        itemStyle: { color: c >= 0 ? 'rgba(245,34,45,0.85)' : 'rgba(22,119,255,0.85)' },
        tooltip: { formatter: () =>
          `B${f.beat_idx} · ${f.label || KP_LABELS[f.name] || f.name} · 贡献 ${(+c).toFixed(2)}` },
      })
    }
    if (data.length) {
      series.push({ type: 'scatter', data, z: 4, emphasis: { focus: 'self' } })
    }
  } else if (attribution && attribution.length && beats.length) {
    // Legacy name-only mode: join a per-name aggregate onto every valid
    // beat's matching keypoint (kept for callers passing feature_points).
    const contribByName = {}
    for (const f of attribution) { if (f && f.name != null) contribByName[f.name] = f.contrib }
    const maxAbs = Math.max(1e-9, ...attribution.map((f) => Math.abs(f.contrib || 0)))
    const data = []
    for (const beat of beats) {
      if (!beat.valid || !beat.keypoints) continue
      for (const [name, sample] of Object.entries(beat.keypoints)) {
        if (!(name in contribByName)) continue
        const contrib = contribByName[name]
        const t = +timeOf(sample).toFixed(4)
        if (t > duration + 0.01) continue
        attribCount += 1
        data.push({
          value: [t, ampAt(t)], name,
          symbolSize: 6 + 14 * (Math.abs(contrib) / maxAbs),
          itemStyle: { color: contrib >= 0 ? 'rgba(245,34,45,0.85)' : 'rgba(22,119,255,0.85)' },
          tooltip: { formatter: () => `${KP_LABELS[name] || name} · 贡献 ${(+contrib).toFixed(2)}` },
        })
      }
    }
    if (data.length) {
      series.push({ type: 'scatter', data, z: 4, emphasis: { focus: 'self' } })
    }
  }

  const option = {
    animation: false,
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    tooltip: { trigger: 'item' },
    xAxis: { type: 'value', min: 0, max: duration,
             name: 't (s)', nameLocation: 'middle', nameGap: 26 },
    yAxis: (yMin != null || yMax != null)
      ? { type: 'value', min: yMin ?? undefined, max: yMax ?? undefined }
      : { type: 'value', scale: true },
    series,
  }

  // ── HTML legend row (ECharts legend can't render markArea swatches)
  const legendItems = []
  if (showValidity && beats.length) {
    legendItems.push(
      <span key="valid" style={legendRowStyle}>
        <span style={{ ...swatchStyle, background: BG_VALID }} /> 有效 beat
      </span>,
      <span key="rejected" style={legendRowStyle}>
        <span style={{ ...swatchStyle, background: BG_REJECTED }} /> 无效 beat
      </span>,
    )
  }
  if (hlCount > 0) {
    legendItems.push(
      <span key="abnormal" style={legendRowStyle}>
        <span style={{ ...swatchStyle, background: BG_ABNORMAL }} /> 异常:{highlightFeature}
      </span>,
    )
  }
  if (showBoundaries && beats.length) {
    legendItems.push(
      <span key="boundary" style={legendRowStyle}>
        <span style={dashSwatchStyle} /> 分割线
      </span>,
    )
  }
  if (attribCount > 0) {
    legendItems.push(
      <span key="attrib" style={legendRowStyle}>
        <span style={{ ...swatchStyle, background: '#f5222d' }} /> 拉高
        <span style={{ ...swatchStyle, background: '#1677ff', marginLeft: 6 }} /> 拉低
        <span>特征点贡献</span>
      </span>,
    )
  }

  return (
    <div>
      {legendItems.length > 0 && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 14,
          padding: '4px 6px 2px', fontSize: 11, color: '#666',
        }}>
          {legendItems}
        </div>
      )}
      <ReactECharts option={option} style={{ height }} notMerge lazyUpdate />
    </div>
  )
}

const legendRowStyle = { display: 'inline-flex', alignItems: 'center', gap: 5 }
const swatchStyle = {
  display: 'inline-block', width: 12, height: 12, borderRadius: 2,
  border: '1px solid rgba(0,0,0,0.15)',
}
const dashSwatchStyle = {
  display: 'inline-block', width: 16, height: 0,
  borderTop: '2px dashed #555555',
}
