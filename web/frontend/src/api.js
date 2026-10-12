// Thin API client — all endpoints go through the Vite proxy (/api → :8000)

async function request(path, options) {
  const resp = await fetch(`/api${path}`, options)
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '')
    throw new Error(`API ${resp.status}: ${detail}`)
  }
  return resp.json()
}

export const api = {
  datasets: () => request('/data/datasets'),
  subjects: (params = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''),
    ).toString()
    return request(`/data/subjects${qs ? `?${qs}` : ''}`)
  },
  subject: (source, subjectId) => request(`/data/subjects/${source}/${subjectId}`),
  waveform: (segId, points = 2000) => request(`/data/segments/${segId}/waveform?points=${points}`),
  // 按 (source,subject,seg_idx) 同步取归一化波形点 + overlay（历史段 Tab 懒加载）
  segmentWaveform: (source, subjectId, segIdx, points = 2000) =>
    request(`/analysis/segment-waveform?source=${encodeURIComponent(source)}`
            + `&subject_id=${encodeURIComponent(subjectId)}&seg_idx=${segIdx}&points=${points}`),
  // L1 可解释性：段级实时预测 + 近窗历史注意力归因
  livePredict: (source, subjectId, segIdx) =>
    request(`/analysis/live-predict?source=${encodeURIComponent(source)}`
            + `&subject_id=${encodeURIComponent(subjectId)}&seg_idx=${segIdx}`),
  // L2 可解释性：当前段波形特征点级 Integrated-Gradients 归因（需 GPU）
  liveAttrib: (source, subjectId, segIdx, target = 'sbp', steps = 32) =>
    request(`/analysis/live-attrib?source=${encodeURIComponent(source)}`
            + `&subject_id=${encodeURIComponent(subjectId)}&seg_idx=${segIdx}`
            + `&target=${target}&steps=${steps}`),
  // 注：L2.5 历史段归因 / L3 解释叙事不再由前端直连，
  // 统一走 Agent 工具 explain_prediction（/chat/message 返回 explain）。
  // 注：异步任务子系统（/tasks + /analysis/process|phenotype + SSE stream）、
  // /reports/* 端点、/adaptation HTTP 路由均已随 Agent 单一控制面重构下线，
  // 前端旧导出同步清除。
  // agent-driven chat analysis
  chatMessage: (payload) => post('/chat/message', payload),
}

function post(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}
