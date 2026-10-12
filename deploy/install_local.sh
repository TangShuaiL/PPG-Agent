#!/usr/bin/env bash
# ── 本机安装【第 1 步：以 <USER> 运行，全程免 sudo】──
# 只做属于 <USER> 家目录的事：放 Caddy 配置 + 生成密钥 env + 停开发态进程。
# 需要 root 的部分（升级 frpc、装 systemd 单元）交给 install_root.sh 以 <ADMIN_USER> 跑。
set -euo pipefail

REPO=<PROJECT_ROOT>
BIN="$HOME/bin"
CFG="$HOME/.config"

[ "$(id -un)" = "<USER>" ] || { echo "!! 请用 <USER> 运行本脚本（当前: $(id -un)）"; exit 1; }

echo "=== 1) 目录 ==="
mkdir -p "$BIN" "$CFG/caddy" "$CFG/ppg"

echo "=== 2) 校验 caddy（本机唯一需下载的二进制）==="
if [ -x "$BIN/caddy" ]; then
  "$BIN/caddy" version | sed 's/^/   caddy /'
else
  echo "!! 缺 $BIN/caddy —— caddyserver API 返回的是裸二进制(非 tar 包), 直接拷:"
  echo '   mkdir -p ~/bin && cp /tmp/caddy.tgz ~/bin/caddy && chmod +x ~/bin/caddy'
  exit 1
fi

echo "=== 3) 放置 Caddyfile ==="
cp "$REPO/deploy/Caddyfile" "$CFG/caddy/Caddyfile"
echo "   -> $CFG/caddy/Caddyfile"

echo "=== 4) 密钥 env（首次生成模板 → 填 key → 重跑）==="
if [ ! -f "$CFG/ppg/env" ]; then
  cat > "$CFG/ppg/env" <<'EOF'
PPG_DATA_ROOT=<DATA_ROOT>/data_v2
DEEPSEEK_API_KEY=REPLACE_ME
EOF
  chmod 600 "$CFG/ppg/env"
  echo "   已生成 $CFG/ppg/env —— 填好 DEEPSEEK_API_KEY 后重跑本脚本。"
  exit 0
fi
grep -q 'REPLACE_ME' "$CFG/ppg/env" && { echo "!! env 里 key 仍是占位符。"; exit 1; }
echo "   env 就绪。"

echo "=== 5) 停开发态进程（<USER> 自己的 uvicorn/vite，免 sudo）==="
pkill -f 'uvicorn web.api.main:app' 2>/dev/null && echo "   已停手动 uvicorn" || echo "   无手动 uvicorn"
pkill -f 'vite' 2>/dev/null && echo "   已停 vite dev" || echo "   无 vite dev"

echo "完成第 1 步。接下来【以 <ADMIN_USER> 运行 install_root.sh】做系统级注册。"
