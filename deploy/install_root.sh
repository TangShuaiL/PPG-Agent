#!/usr/bin/env bash
# ── 本机安装【第 2 步：以 <ADMIN_USER> 运行，需要 sudo】──
# 只做需要 root 的系统级注册：升级 /opt/frp/frpc、写 /opt/frp/frpc.toml、
# 安装并启用 systemd 单元（服务本身仍以 User=<USER> 运行）。
# 前置：先以 <USER> 跑完 install_local.sh（caddy/env/Caddyfile 就位）。
set -euo pipefail

REPO=<PROJECT_ROOT>
FRPC_SRC=/tmp/frp_0.71.0_linux_amd64/frpc

[ "$(id -un)" = "<ADMIN_USER>" ] || { echo "!! 请用 <ADMIN_USER> 运行本脚本（当前: $(id -un)）"; exit 1; }
echo "=== 0) 前置校验 ==="
[ -x "$FRPC_SRC" ] || { echo "!! 未找到 $FRPC_SRC（0.71 frpc 未解压）"; exit 1; }
[ -f "$REPO/deploy/frpc.toml" ] || { echo "!! 缺 $REPO/deploy/frpc.toml"; exit 1; }
# 注: <USER> 的 ~/.config 是 0700, <ADMIN_USER> 无 sudo 进不去 → 用 sudo 探测(root 可读)。
sudo test -f $XDG_CONFIG_HOME/caddy/Caddyfile || { echo "!! 缺 <USER> 的 Caddyfile —— 先以 <USER> 跑 install_local.sh"; exit 1; }
sudo test -f $XDG_CONFIG_HOME/ppg/env && ! sudo grep -q REPLACE_ME $XDG_CONFIG_HOME/ppg/env || { echo "!! <USER> 的 env 未就绪/key 未填 —— 先完成 install_local.sh"; exit 1; }
echo "   前置 OK。"

echo "=== 1) 停 frpc 并升级到 0.71（避开'文本文件忙'）==="
sudo systemctl stop frpc
sudo cp "$FRPC_SRC" /opt/frp/frpc && sudo chmod 755 /opt/frp/frpc
/opt/frp/frpc --version | sed 's/^/   frpc /'

echo "=== 2) 写隧道配置（web-only + 新 token + TLS，删旧 ssh 代理）==="
sudo cp -n /opt/frp/frpc.toml /opt/frp/frpc.toml.bak 2>/dev/null || true
sudo cp "$REPO/deploy/frpc.toml" /opt/frp/frpc.toml

echo "=== 3) 安装 systemd 单元（ppg-api / ppg-caddy 以 User=<USER> 运行）==="
sudo cp "$REPO/deploy/systemd/ppg-api.service" \
        "$REPO/deploy/systemd/ppg-caddy.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo "=== 4) 启用并启动三个服务 ==="
sudo systemctl enable --now frpc
sudo systemctl enable --now ppg-api ppg-caddy
sleep 3

echo "=== 5) 状态回显 ==="
for s in frpc ppg-api ppg-caddy; do printf '   %-12s %s\n' "$s" "$(systemctl is-active $s)"; done
echo "   本机自测（以 <USER> 身份 curl 回环 8088）:"
curl -s -o /dev/null -w "     Caddy / -> http=%{http_code}\n" http://127.0.0.1:8088/ || true
curl -s -o /dev/null -w "     Caddy /api/health -> http=%{http_code}\n" http://127.0.0.1:8088/api/health || true
echo "完成。/api/health=200 即本机就绪，隧道会把 8088 透到云机。"
