# -*- coding: utf-8 -*-
"""eval/run_campaign.py — 战役编排器：一份 yaml 定义整场战役，CLI 差量传参。

配置层级（2026-09-05 定稿）:
  预训练   : config/base/<模型>.yaml 基座 + sh 传参 override（train/pretrain.py load_config）
  下游探针 : 各入口命令行参数（scripts/ 下 extract_embeddings / materialize_stream /
             probe_full_stream，均已参数化）
  战役层   : config/campaign_*.yaml（本脚本消费）+ CLI 差量 —— 只有一份，不按实验膨胀

用法:
  python eval/run_campaign.py config/campaign_example.yaml --dry-run     # 只打印命令
  python eval/run_campaign.py config/campaign_example.yaml               # 全量执行
  python eval/run_campaign.py config/campaign_example.yaml --only beatmae_v2__v2c_4src_mvaiwb__full
  python eval/run_campaign.py config/campaign_example.yaml --tasks extract,stream
  python eval/run_campaign.py config/campaign_example.yaml --corpus /path/to/corpus
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def build_commands(cfg, exps, tasks):
    """按阶段顺序生成 shell 命令列表（cwd=仓库根）。

    每实验可用 tasks: [...] 覆盖全局阶段；stream: true 表示该实验需要流式评估
    （先 materialize_stream 物化，再 probe_full_stream 跑变体，物化目录=STREAM_CACHE/<exp>）。
    """
    py = cfg.get('python') or sys.executable
    cache_dir = cfg.get('cache_dir', 'results/cache/embeddings')
    stream_cache = cfg.get('stream_cache', 'results/cache/stream')
    # 变体子集透传（probe_full_stream.py 吃位置参数，缺省=stream_probe.VARIANTS 主表三变体）。
    # 2026-09-11 接通：此前 yaml 里的 stream_variants 没人读，是个改了不生效的假旋钮。
    sv = cfg.get('stream_variants') or []
    sv_arg = (' ' + ' '.join(sv)) if sv else ''
    cmds = []
    for name, e in exps.items():
        src = e.get('sources')
        src_arg = f' --sources={",".join(src)}' if src else ''
        kind_arg = f' --kind={e["kind"]}' if e.get('kind') else ''
        pl_arg = f' --patch-len={e["patch_len"]}' if e.get('patch_len') else ''
        # 波形口径透传（raw_waveform 与 patch_len 配套，见 extract_embeddings.py 顶部）。
        # 2026-09-11 接通：与 stream_variants 同病——yaml 写了没人读，改了不生效。
        im_arg = f' --input-mode={e["input_mode"]}' if e.get('input_mode') else ''
        cache = e.get('cache') or f'{cache_dir}/{name}_lab.npz'
        eff = e.get('tasks', tasks)   # per-experiment 阶段覆盖
        if 'extract' in eff:
            if Path(cache).exists():   # 幂等：缓存已在则跳过提取，防覆盖现有 npz
                print(f'[campaign] 跳过提取（缓存已存在）: {cache}', flush=True)
            else:
                cmds.append(f'{py} scripts/extract_embeddings.py --name={name} --ckpt={e["ckpt"]}{src_arg}{kind_arg}{pl_arg}{im_arg}')
        if 'window' in eff or e.get('window'):
            # window 阶段已退役（2026-09-10，26 号归档）。**显式报错，不静默跳过**：
            # yaml 里还写着 window 的实验必须看得见，否则会以为跑过了。
            sys.exit(f'✗ {name}: 请求了已退役的 window 阶段。\n'
                     f'  本项目最终口径为 segment 段级流式，请改用该实验的 stream: true'
                     f'（materialize_stream + probe_full_stream）。')
        if e.get('stream'):
            cmds.append(f'{py} scripts/materialize_stream.py --exp={name} --npz={cache}')
            cmds.append(f'{py} scripts/probe_full_stream.py exp={name} dir={stream_cache}/{name}{sv_arg}')
    return cmds


def main():
    ap = argparse.ArgumentParser(description='战役编排器（yaml 驱动 + CLI 差量）')
    ap.add_argument('config', help='config/campaign_*.yaml')
    ap.add_argument('--only', help='逗号分隔实验名子集')
    ap.add_argument('--tasks', help='逗号分隔阶段子集（默认按 yaml tasks 顺序）')
    ap.add_argument('--corpus', help='覆盖 corpus_root（写入 PPG_CORPUS_ROOT 传给子进程）')
    ap.add_argument('--dry-run', action='store_true', help='只打印将执行的命令')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    exps = {k: v for k, v in cfg['experiments'].items()
            if not a.only or k in a.only.split(',')}
    if not exps:
        sys.exit(f'--only={a.only} 未匹配任何实验: {list(cfg["experiments"])}')
    tasks = (a.tasks or ','.join(cfg['tasks'])).split(',')
    unknown = set(tasks) - {'extract', 'window', 'stream'}
    if unknown:
        sys.exit(f'未知阶段: {unknown}（可选 extract/stream）')
    if 'window' in tasks:      # 2026-09-10 退役（见 build_commands 注释）
        sys.exit('✗ window 阶段已退役（26 号归档 2026-09-10）；段级流式请用 stream')

    env = dict(os.environ)
    corpus = a.corpus or cfg.get('corpus_root')
    if corpus:
        env['PPG_CORPUS_ROOT'] = corpus
        print(f'[campaign] PPG_CORPUS_ROOT={corpus}', flush=True)

    cmds = build_commands(cfg, exps, tasks)
    print(f'战役 {a.config}: {len(exps)} 实验 × 阶段 {tasks} → {len(cmds)} 条命令', flush=True)
    for c in cmds:
        print(f'  $ {c}', flush=True)
    if a.dry_run:
        print('dry-run 结束，未执行。', flush=True)
        return
    for c in cmds:
        r = subprocess.run(c, shell=True, cwd=REPO, env=env)
        if r.returncode != 0:
            sys.exit(f'✗ 命令失败（exit {r.returncode}）: {c}')
    print('══ 战役完成 ══', flush=True)


if __name__ == '__main__':
    main()
