import React from 'react'
import { Layout, Menu } from 'antd'
import {
  DatabaseOutlined, MessageOutlined,
} from '@ant-design/icons'
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import ExplorerPage from './pages/ExplorerPage.jsx'
import ChatAnalysisPage from './pages/ChatAnalysisPage.jsx'

const { Header, Content, Sider } = Layout

const MENU = [
  { key: '/explorer', icon: <DatabaseOutlined />, label: '数据浏览器' },
  { key: '/analysis', icon: <MessageOutlined />, label: '分析对话' },
]

function AppLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const selected = MENU.find((m) => location.pathname.startsWith(m.key))?.key
  return (
    <Layout style={{ height: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img src="/ppg-agent-logo.png" alt="PPG-Agent"
               style={{ height: 32, width: 32, borderRadius: 6, marginRight: 10 }} />
          <div style={{ color: '#fff', fontSize: 18, fontWeight: 600 }}>
            PPG-Agent <span style={{ fontSize: 13, opacity: 0.7 }}>心血管表型分析</span>
          </div>
        </div>
      </Header>
      <Layout style={{ flex: 1, minHeight: 0 }}>
        <Sider width={200} theme="light">
          <Menu
            mode="inline"
            selectedKeys={[selected]}
            items={MENU}
            onClick={({ key }) => navigate(key)}
            style={{ height: '100%', borderRight: 0, paddingTop: 8 }}
          />
        </Sider>
        <Content style={{ padding: 16, background: '#f5f5f5', overflow: 'auto',
                          position: 'relative' }}>
          <Routes>
            <Route path="/" element={<Navigate to="/explorer" replace />} />
            <Route path="/explorer" element={<ExplorerPage />} />
            <Route path="/analysis" element={<ChatAnalysisPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  )
}
