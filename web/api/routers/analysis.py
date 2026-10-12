"""Analysis read endpoints: segment waveform + L1/L2 live explainability.

The async task-submission pipeline (/process, /phenotype) and the L2.5/L3
direct-explain endpoints (/live-attrib-history, /live-explain) were retired
with the Agent single-control-plane refactor — the frontend now drives
history/explanation through the chat Agent (explain_prediction).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/segment-waveform")
def segment_waveform(source: str = Query(..., description="vital|mimic|aurora|anker"),
                     subject_id: str = Query(...),
                     seg_idx: int = Query(..., ge=0),
                     points: int = Query(default=2000, ge=200, le=5000)):
    """按 (source, subject_id, seg_idx) 同步取归一化波形点 + beat overlay。

    供对话页历史段 Tab 懒加载：在同一固定坐标轴 (x:[0,10s], y:[0,1]) 上渲染
    历史片段轨迹并叠加关键点。库内波形已 min-max 归一化到 [0,1]、500Hz/10s
    (Aurora 半长窗在加载时 resample 到 5000)。纯引擎链（无 GPU）；段缺失返回
    404。overlay 关键点为绝对 500Hz 采样坐标（与 WaveformChart 对齐口径一致）。
    """
    from ppg_agent.context import reset_context, get_context
    from ..services import analysis_service as svc
    from ..services.waveform_reader import downsample
    try:
        reset_context()  # 请求级隔离：装一个全新 PPGContext
        svc.load_signal(source, subject_id, seg_idx)
        # 幂等归一化到 [0,1]（库内已归一则 no-op）：保证历史轨迹与
        # 当前段同 y 域，固定坐标轴不跨段漂移。
        try:
            svc.normalize()
        except ValueError:
            pass  # 平坦段(零量程)不归一化，直接过
        beats = svc.segment_beats_step()
        overlay = svc.beat_overlay()
    except KeyError as e:
        raise HTTPException(404, f"segment not found: {source}/{subject_id}/{seg_idx}: {e}")
    except FileNotFoundError as e:
        raise HTTPException(404, f"segment not found: {source}/{subject_id}/{seg_idx}: {e}")
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001 — h5py/scipy/engine 失败
        raise HTTPException(500, f"failed to build segment waveform: {e}")
    ctx = get_context()
    n = min(max(int(points), 200), 5000)
    return {
        "source": source, "subject_id": subject_id, "seg_idx": seg_idx,
        "fs": int(ctx.fs),
        "n_samples": int(len(ctx.ppg_signal)),
        "duration_sec": float(ctx.signal_ref.duration_sec),
        "points": downsample(ctx.ppg_signal, n),
        "overlay": overlay,
        "beats": beats,
    }


@router.get("/live-predict")
def live_predict(source: str = Query(..., description="vital|mimic|aurora|anker"),
                 subject_id: str = Query(...),
                 seg_idx: int = Query(..., ge=0)):
    """L1 可解释性：段级预测 + 近窗注意力归因。

    返回 ``predictions`` (sbp/dbp/age) 与 ``history_attention``（当前段本次
    估计参考了哪些前序历史段，按 target × seed×折平均，仅覆盖近 K 段），
    附 ``measured_bp`` 实测标签对照。主口径（实时编码）不可用时降级查表
    （同头同嵌入，逐位等价）。模型输出为群体统计估计，非诊断结论。"""
    from ppg_agent.engine.causal_stream import get_causal_stream_engine
    from ppg_agent.engine.live_stream import (LiveStreamUnavailable,
                                               get_live_stream_engine)
    try:
        try:
            eng = get_live_stream_engine()
            out = eng.predict_segment_stream(source, subject_id, seg_idx)
        except LiveStreamUnavailable as e:
            out = get_causal_stream_engine().predict_segment_lookup(
                source, subject_id, seg_idx)
            out["live_unavailable_reason"] = str(e)
        try:
            out["measured_bp"] = get_live_stream_engine().measured_bp(
                source, subject_id, seg_idx)
        except Exception:  # noqa: BLE001 — 无标签/读失败不影响归因主体
            out["measured_bp"] = None
    except KeyError as e:
        raise HTTPException(404, str(e))
    return out


@router.get("/live-attrib")
def live_attrib(source: str = Query(..., description="vital|mimic|aurora|anker"),
               subject_id: str = Query(...),
               seg_idx: int = Query(..., ge=0),
               target: str = Query(default="sbp", description="sbp|dbp|age"),
               steps: int = Query(default=32, ge=8, le=128)):
    """L2 可解释性：当前段波形的特征点级 Integrated-Gradients 归因。

    返回 ``feature_points``（按重要度排序的 9 个生理关键点：onset/收缩峰/
    重搏切迹/潮波… 的签名贡献，跨全段聚合）与 ``beat_points``（逐 (拍, 关键点)
    的签名贡献 + 绝对时间(秒)，供前端把梯度归因定位到具体拍的具體关键点，而非
    每个 beat 同色高亮）及 ``beats_with_keypoints`` 等元数据，附
    ``completeness_gap``（IG 自校验：∑归因 ≈ 实际预测 − baseline 预测，应近 0）。
    需 GPU+encoder（实时链）；不可用时返回 503。输出为群体模型输入
    依赖度，非生理病因，不作诊断。"""
    from ppg_agent.engine.live_stream import (LiveStreamUnavailable,
                                               get_live_stream_engine)
    if target not in ("sbp", "dbp", "age"):
        raise HTTPException(400, "target 必须为 sbp|dbp|age")
    try:
        return get_live_stream_engine().segment_waveform_attribution(
            source, subject_id, seg_idx, target, steps=steps)
    except LiveStreamUnavailable as e:
        raise HTTPException(503, f"实时归因链不可用（需 GPU+encoder）: {e}")
    except KeyError as e:
        raise HTTPException(404, str(e))

