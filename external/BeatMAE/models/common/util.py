"""util：与具体架构无关的小工具（冒烟测试与训练脚本共用）。"""


def n_params(m):
    """模型参数量（原始个数；要打印成 M 自行 /1e6）。"""
    return sum(p.numel() for p in m.parameters())
