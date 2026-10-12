#!/usr/bin/env python3
"""色盲模拟 + CIEDE2000，用于校验论文配色的可辨性。

用途：线性图/柱状图里任意两条系列都可能被并排比较，所以要验**全配对**而不只相邻对。
判据（沿用本项目既有口径）：各色觉条件下 ΔE 最差 ≥ 15；色 vs 白底对比度 ≥ 3:1。
"""
import numpy as np

M_HPE = np.array([[17.8824, 43.5161, 4.11935],
                  [3.45565, 27.1554, 3.86714],
                  [0.0299566, 0.184309, 1.46709]])   # 线性 RGB → LMS
_INV = np.linalg.inv(M_HPE)
_PROJ = {  # LMS 空间下的缺失锥细胞投影（Viénot 1999 / Brettel 1997）
    'protan': np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
    'deutan': np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
    'tritan': np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
}
SIM = {k: _INV @ P @ M_HPE for k, P in _PROJ.items()}
SIM['normal'] = np.eye(3)


def _hex2rgb(h):
    if isinstance(h, str):
        h = h.lstrip('#')
        return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.asarray(h, float)


def _srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(hexcol, kind):
    return _lin2srgb(SIM[kind] @ _srgb2lin(_hex2rgb(hexcol)))


def _lab_from_xyz(xyz):
    xyz = np.asarray(xyz, float)
    w = np.array([0.95047, 1.0, 1.08883])
    t = xyz / w
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def hex2lab(h):
    lin = _srgb2lin(_hex2rgb(h))
    xyz = lin @ np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]]).T
    return _lab_from_xyz(xyz)


def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1; L2, a2, b2 = lab2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(abs(h1p - h2p) > 180, (hsum + 360) / 2,
                   np.where(hsum < 360, hsum / 2, (hsum - 360) / 2))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    RC = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    SL = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dth)) * RC
    return float(np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                         + RT * (dCp / SC) * (dHp / SH)))


def rel_lum(h):
    lin = _srgb2lin(_hex2rgb(h))
    return float(lin @ [0.2126, 0.7152, 0.0722])


def contrast_vs_white(h):
    return (1.0 + 0.05) / (rel_lum(h) + 0.05)


def de_under(hex1, hex2, kind):
    """在指定色觉条件下两色的 ΔE（先模拟再算）。"""
    return ciede2000(hex2lab(simulate(hex1, kind)), hex2lab(simulate(hex2, kind)))


def report(palette, names=None, min_de=15.0, min_cr=3.0):
    names = names or list(palette)
    cols = [palette[n] for n in names]
    print('  色：' + '  '.join(f'{n}={c}' for n, c in zip(names, cols)))
    worst = {}
    for kind in ('normal', 'protan', 'deutan', 'tritan'):
        vals = [(de_under(cols[i], cols[j], kind), names[i], names[j])
                for i in range(len(cols)) for j in range(i + 1, len(cols))]
        vals.sort()
        worst[kind] = vals[0][0]
        flag = 'PASS' if vals[0][0] >= min_de else 'FAIL'
        detail = ' | '.join(f'{a}↔{b} {v:.1f}' for v, a, b in vals[:3])
        print(f'  {kind:<6} 最差 ΔE {vals[0][0]:5.1f}  [{flag}]   {detail}')
    crs = [(contrast_vs_white(c), n) for n, c in zip(names, cols)]
    crs.sort()
    cf = 'PASS' if crs[0][0] >= min_cr else 'FAIL'
    print(f'  对比度(vs 白) 最低 {crs[0][0]:.2f}:1 [{cf}]  ({crs[0][1]})')
    ok = all(v >= min_de for v in worst.values()) and crs[0][0] >= min_cr
    print(f'  ⇒ 综合 {"PASS" if ok else "FAIL"}')
    return ok, worst, crs


def main():
    """校验 `_paper_style.py` 里实际使用的调色板（**从该模块导入，不在此复制**）。

    判据只对**分类配色**成立 —— 序列型色阶（单色由浅到深编码大小）本就该相邻接近，
    不能套用本条。故此处只验 TARGET。

    直接运行：python results/paper/code/_cvd.py
    退出码 0 = 全部 PASS，1 = 有 FAIL。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paper_style import TARGET       # 调色板的唯一来源

    print('目标三色（数据系列 sbp / dbp / age）：')
    ok, _, _ = report(TARGET)

    print(f'\n{"全部通过" if ok else "存在未达标项"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
