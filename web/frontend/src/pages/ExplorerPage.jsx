import React, { useEffect, useState, useCallback } from 'react'
import {
  Row, Col, Card, Statistic, Table, Select, Tag, Drawer,
  Space, Descriptions, Spin, message, Modal, Upload, Button,
} from 'antd'
import { UploadOutlined, InboxOutlined } from '@ant-design/icons'
import { api } from '../api.js'
import WaveformChart from '../components/WaveformChart.jsx'
import { SOURCE_COLORS, TIER_COLORS } from '../components/waveTheme.js'

function UploadModal({ open, onClose, onUploaded }) {
  const [uploading, setUploading] = useState(false)
  const [source, setSource] = useState('custom')

  const props = {
    name: 'file',
    multiple: false,
    showUploadList: false,
    beforeUpload: (file) => {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('source', source)
      setUploading(true)
      fetch('/api/data/upload', {
        method: 'POST',
        body: fd,
      })
        .then((r) => (r.ok ? r.json() : r.text().then((t) => { throw new Error(t) })))
        .then((info) => {
          message.success(`已保存: ${info.stored} (${(info.size_bytes / 1024).toFixed(1)} KB)`)
          onUploaded?.(info)
          onClose()
        })
        .catch((e) => message.error(`上传失败: ${e.message}`))
        .finally(() => setUploading(false))
      return false // handled manually
    },
  }

  return (
    <Modal title="上传数据" open={open} onCancel={onClose} footer={null} width={480}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Select value={source} style={{ width: 200 }}
                onChange={setSource}
                options={['custom', 'vital', 'mimic', 'aurora']
                  .map((s) => ({ value: s, label: s === 'custom' ? '自定义' : s.toUpperCase() }))} />
        <Upload.Dragger {...props} disabled={uploading}>
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
          <p className="ant-upload-hint">支持 .mat / .h5 / .csv / .txt 原始波形文件（后续接入入库流程）</p>
        </Upload.Dragger>
        {uploading && <Spin size="small" />}
      </Space>
    </Modal>
  )
}

function DatasetCards({ datasets }) {
  return (
    <Row gutter={16}>
      {datasets.map((d) => (
        <Col span={8} key={d.source}>
          <Card size="small">
            <Statistic
              title={<Tag color={SOURCE_COLORS[d.source]}>{d.source.toUpperCase()}</Tag>}
              value={d.subjects}
              suffix="受试者"
            />
            <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
              segments: {Object.entries(d.segments || {}).map(([tier, n]) =>
                <Tag key={tier} color={TIER_COLORS[tier]}>{tier}: {n.toLocaleString()}</Tag>)}
              {d.scenarios?.a != null && (
                <span style={{ marginLeft: 8 }}>
                  <Tag>实验室 a: {d.scenarios.a}</Tag>
                  <Tag>动态 o: {d.scenarios.o || 0}</Tag>
                </span>
              )}
            </div>
          </Card>
        </Col>
      ))}
    </Row>
  )
}

function SubjectDetail({ source, subjectId, onClose }) {
  const [detail, setDetail] = useState(null)
  const [waveform, setWaveform] = useState(null)
  const [activeSeg, setActiveSeg] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!subjectId) return
    api.subject(source, subjectId).then(setDetail).catch((e) => message.error(e.message))
  }, [source, subjectId])

  const showWaveform = useCallback(async (seg) => {
    setActiveSeg(seg.id)
    setWaveform(null)
    setLoading(true)
    try {
      setWaveform(await api.waveform(seg.id, 2000))
    } catch (e) {
      message.error(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const segColumns = [
    { title: 'seg_idx', dataIndex: 'seg_idx', width: 80 },
    {
      title: '质量', dataIndex: 'quality_tier', width: 90,
      render: (t, r) => <Tag color={TIER_COLORS[t]}>{t} {(r.quality_score ?? 0).toFixed(2)}</Tag>,
    },
    { title: '有效/总 beats', width: 110, render: (_, r) => `${r.n_beats_valid}/${r.n_beats_total}` },
    { title: 'SBP', dataIndex: 'sbp_label', width: 70, render: (v) => v ?? '—' },
    { title: 'DBP', dataIndex: 'dbp_label', width: 70, render: (v) => v ?? '—' },
  ]

  return (
    <Drawer title={`${source} / ${subjectId}`} open={!!subjectId} onClose={onClose} width={720}>
      {detail && (
        <>
          <Descriptions size="small" column={4} bordered style={{ marginBottom: 16 }}>
            <Descriptions.Item label="年龄">{detail.subject.age ?? '—'}</Descriptions.Item>
            <Descriptions.Item label="性别">{detail.subject.gender ?? '—'}</Descriptions.Item>
            <Descriptions.Item label="场景">{detail.subject.scenario ?? '—'}</Descriptions.Item>
            <Descriptions.Item label="划分">{detail.subject.split}</Descriptions.Item>
          </Descriptions>
          <Table
            size="small" rowKey="id" columns={segColumns}
            dataSource={detail.segments} pagination={{ pageSize: 8 }}
            onRow={(seg) => ({ onClick: () => showWaveform(seg), style: { cursor: 'pointer' } })}
            rowClassName={(seg) => (seg.id === activeSeg ? 'ant-table-row-selected' : '')}
          />
          <Card size="small" title={activeSeg ? `Segment #${activeSeg} 波形 (500 Hz)` : '点击 segment 查看波形'}
                style={{ marginTop: 16 }}>
            {loading && <Spin />}
            {waveform && !loading && (
              <>
                <WaveformChart points={waveform.points} fs={waveform.fs}
                               durationSec={waveform.duration_sec} />
                <div style={{ fontSize: 12, color: '#666' }}>
                  时长 {waveform.duration_sec}s · 原始采样点 {waveform.n_samples} ·
                  幅值 [{waveform.stats.min}, {waveform.stats.max}]
                </div>
              </>
            )}
          </Card>
        </>
      )}
    </Drawer>
  )
}

export default function ExplorerPage() {
  const [datasets, setDatasets] = useState([])
  const [filters, setFilters] = useState({ source: 'vital', scenario: '', gender: '', split: '' })
  const [page, setPage] = useState(1)
  const [data, setData] = useState({ items: [], total: 0 })
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null) // {source, subject_id}
  const [uploadOpen, setUploadOpen] = useState(false)

  useEffect(() => { api.datasets().then(setDatasets).catch((e) => message.error(e.message)) }, [])

  useEffect(() => {
    setLoading(true)
    api.subjects({ ...filters, page, page_size: 20 })
      .then(setData)
      .catch((e) => message.error(e.message))
      .finally(() => setLoading(false))
  }, [filters, page])

  const columns = [
    { title: '受试者', dataIndex: 'subject_id' },
    {
      title: '数据集', dataIndex: 'source',
      render: (s) => <Tag color={SOURCE_COLORS[s]}>{s}</Tag>,
    },
    { title: '场景', dataIndex: 'scenario', render: (v) => v || '—' },
    { title: '年龄', dataIndex: 'age', render: (v) => v?.toFixed(0) ?? '—' },
    { title: '性别', dataIndex: 'gender' },
    { title: 'segments', dataIndex: 'n_segments' },
    { title: '划分', dataIndex: 'split', render: (v) => <Tag>{v}</Tag> },
  ]

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row justify="end" style={{ width: '100%' }}>
        <Button type="primary" icon={<UploadOutlined />}
                onClick={() => setUploadOpen(true)}>
          上传数据
        </Button>
      </Row>
      <DatasetCards datasets={datasets} />
      <Card size="small">
        <Space wrap>
          <Select value={filters.source} style={{ width: 120 }}
                  onChange={(v) => { setPage(1); setFilters((f) => ({ ...f, source: v, scenario: '' })) }}
                  options={['vital', 'mimic', 'aurora'].map((s) => ({ value: s, label: s.toUpperCase() }))} />
          {filters.source === 'aurora' && (
            <Select value={filters.scenario} style={{ width: 150 }} allowClear placeholder="场景"
                    onChange={(v) => { setPage(1); setFilters((f) => ({ ...f, scenario: v || '' })) }}
                    options={[{ value: 'a', label: '实验室 (a)' }, { value: 'o', label: '动态 (o)' }]} />
          )}
          <Select value={filters.gender} style={{ width: 100 }} allowClear placeholder="性别"
                  onChange={(v) => { setPage(1); setFilters((f) => ({ ...f, gender: v || '' })) }}
                  options={[{ value: 'M', label: '男' }, { value: 'F', label: '女' }]} />
          <Select value={filters.split} style={{ width: 110 }} allowClear placeholder="划分"
                  onChange={(v) => { setPage(1); setFilters((f) => ({ ...f, split: v || '' })) }}
                  options={[{ value: 'train', label: 'train' }, { value: 'test', label: 'test' }]} />
        </Space>
      </Card>
      <Card size="small">
        <Table
          size="small" rowKey={(r) => `${r.source}/${r.subject_id}`}
          columns={columns} dataSource={data.items} loading={loading}
          pagination={{
            current: page, pageSize: 20, total: data.total, showTotal: (t) => `共 ${t.toLocaleString()} 人`,
            onChange: setPage,
          }}
          onRow={(r) => ({
            onClick: () => setSelected({ source: r.source, subject_id: r.subject_id }),
            style: { cursor: 'pointer' },
          })}
        />
      </Card>
      <SubjectDetail
        source={selected?.source} subjectId={selected?.subject_id}
        onClose={() => setSelected(null)}
      />
      <UploadModal open={uploadOpen} onClose={() => setUploadOpen(false)} />
    </Space>
  )
}
