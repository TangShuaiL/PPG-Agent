// Unified waveform visualization theme — one source of truth for all
// waveform annotation colors across modes (quality / phenotype / adapt).
//
// Validity tints + boundary style mirror the matplotlib reference
// (ppg_agent/engine/visualization.py: BG_VALID / BG_REJECTED / '#555555')
// so the ECharts version matches the legacy PNG figures.

// Fiducial-point colors (Aurora keypoint naming convention)
export const KP_COLORS = {
  o: '#52c41a', s: '#1677ff', md: '#13c2c2', b: '#722ed1',
  dn: '#fa8c16', ip: '#eb2f96', d: '#f5222d', a: '#a0d911', v: '#8c8c8c',
}

// 特征点名 (feature-point display names, compact for on-chart labels)
export const KP_LABELS = {
  o: 'o 起始', a: 'a a波', md: 'md 最大斜率', b: 'b 斜坡', s: 's 收缩峰',
  dn: 'dn 重搏切迹', ip: 'ip 拐点', d: 'd 舒张峰', v: 'v 结束',
}

// Beat backgrounds — matplotlib parity
export const BG_VALID = '#d5f5e3'     // light green
export const BG_REJECTED = '#fadbd8'  // light red
export const BG_ABNORMAL = '#ffd591'  // amber — abnormal-feature highlight
export const BOUNDARY_COLOR = '#555555'

// Engine level vocabulary (stratified_percentiles / per-beat feats):
// very_high / high / mildly_high / normal / mildly_low / low / very_low
export const LEVEL_COLORS = {
  very_high: 'red', high: 'volcano', mildly_high: 'orange',
  moderate: 'orange', // legacy engine level
  mildly_low: 'lime', low: 'green', very_low: 'green', normal: 'green',
}
export const LEVEL_ABNORMAL = new Set([
  'very_high', 'high', 'mildly_high', 'low', 'very_low', 'mildly_low',
])

export const TIER_COLORS = { good: 'green', fair: 'orange', failed: 'red' }
export const SOURCE_COLORS = { vital: 'green', mimic: 'blue', aurora: 'volcano' }
