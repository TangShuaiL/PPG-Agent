#!/usr/bin/env python3
"""run_meta.py — 运行记录统一出口：元数据 + 日志 + ckpt 身份。

复现性记录的唯一来源，三件事：
- write_meta()：所有入口脚本（pretrain / pretrain_baseline / probe）启动时
  落盘 run_meta.json；预训练 ckpt 另存轻量 meta（commit + 环境 + 配置路径，
  cfg 已单独存于 ckpt 不重复），下游 probe / 回溯只需权重文件即可追溯来源。
- log()：控制台 + train.log/probe.log 双写（各脚本内的 `_log = lambda ...`
  惯用法即本函数的偏应用）。
- ckpt_sha1()：权重文件内容指纹，经 encoder_version 写进嵌入产物，
  换模型时据此重算。

run_meta.json 结构：
{
  "created_at":  启动时刻 ISO 格式,
  "config_path": 使用的配置文件路径（覆盖前）,
  "git":         {"commit", "branch", "dirty"},
  "host":        {"hostname", "python", "torch", "cuda", "gpu"},
  "cfg":         应用 overrides 后的生效配置（完整快照）
}
"""
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────── 通用出口 ──────────────────────────
def log(msg, log_path=None):
    """打印到控制台并（可选）追加进日志文件。各入口脚本的运行记录出口。"""
    print(msg, flush=True)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(msg + '\n')


def ckpt_sha1(path) -> str:
    """权重文件内容 sha1（前 12 位）——ckpt 身份指纹，串进 encoder_version。"""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def _git_info(root):
    """读取当前 git 状态；非 git 仓库 / 无提交 / git 异常时字段为 None（不阻断训练）。"""
    info = {'commit': None, 'branch': None, 'dirty': None}

    def run(*cmd):
        r = subprocess.run(('git',) + cmd, cwd=root, capture_output=True,
                           text=True, timeout=10)
        return r.returncode, r.stdout.strip()

    try:
        rc, inside = run('rev-parse', '--is-inside-work-tree')
        if rc == 0 and inside == 'true':
            rc, commit = run('rev-parse', 'HEAD')
            info['commit'] = commit if rc == 0 and len(commit) == 40 else None
            rc, branch = run('rev-parse', '--abbrev-ref', 'HEAD')
            info['branch'] = branch if rc == 0 else None
            rc, status = run('status', '--porcelain')
            info['dirty'] = bool(status) if rc == 0 else None
    except Exception:
        pass
    return info


def _host_info():
    d = {'hostname': socket.gethostname(), 'python': sys.version.split()[0]}
    try:
        import torch
        d['torch'] = torch.__version__
        d['cuda'] = torch.version.cuda
        d['gpu'] = (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else 'cpu')
    except Exception:
        pass
    return d


def collect_meta(cfg, config_path=None):
    """组装完整 meta（含 cfg 快照）。"""
    return {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'config_path': str(config_path) if config_path else None,
        'git': _git_info(_REPO_ROOT),
        'host': _host_info(),
        'cfg': cfg,
    }


def write_meta(out_dir, cfg, config_path=None):
    """落盘 run_meta.json 并返回完整 meta（供入口脚本复用）。"""
    meta = collect_meta(cfg, config_path)
    p = Path(out_dir) / 'run_meta.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    return meta


def ckpt_meta(meta):
    """ckpt 用轻量版：去掉 cfg 快照（预训练 ckpt 已单独存 cfg，避免重复）。"""
    return {k: v for k, v in meta.items() if k != 'cfg'}
