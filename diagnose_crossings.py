#!/usr/bin/env python3
"""
诊断工具：在 perticle_1_drop.mov 中找到
  - 计时器归零重置帧（粒子在0刻度线 = 自由落体起点）
  - 粒子越过1.6刻度线的帧
并保存带标注的截图到 /tmp/。

检测策略：
  计时器每帧都在跳动，均值差分无法区分"正常计数"和"归零重置"。
  改用"冻结检测"：计时器冻结时连续帧的局部方差极低；
  解冻（归零重置）时方差突然升高。
  在计时器数字区域 (x≈40-100, y≈25-55 的 ROI 局部坐标) 计算滚动方差。

用法：
    python3 diagnose_crossings.py
"""

import cv2
import numpy as np
import os

# ── 参数 ─────────────────────────────────────────────────────────────────────
VIDEO           = "perticle_1_drop.mov"
FPS             = 10.0

# 计时器完整 ROI（右上角）
TIMER_ROI       = (1050, 0, 230, 80)   # (x, y, w, h)
# 计时器数字子区域（ROI 局部坐标）
DIGIT_SUBX      = slice(30, 140)
DIGIT_SUBY      = slice(22, 55)

# 冻结检测参数
FREEZE_WIN      = 5     # 冻结判定窗口（连续帧数）
FREEZE_THR_LOW  = 6.0   # 冻结期帧间差分上界
FREEZE_THR_HIGH = 8.0   # 解冻判定阈值
SCAN_FRAMES     = 300

# 刻度线参数
ZERO_LINE_Y     = 54.0
LINE_16_Y       = 54.0 + 1.6 * 84.0   # 188.4 px

# 粒子检测参数
SEARCH_BAND     = 40
X_HINT          = 665   # 已知粒子 x 坐标（来自历史分析）
X_RANGE         = 100
X_MAX           = 950   # 排除右侧显示区域
X_DRIFT_MAX     = 180   # 跟踪时允许的最大 x 漂移量（防止跳变到其他粒子）
MAX_DY_PER_FRAME = 3.5  # 速度门限：每帧最大 y 位移 (px)，目标粒子≈0.5px/帧，快速粒子≈8px/帧
MIN_BLOB_VAL    = 8
BG_FRAMES       = 55

# !! 硬编码 onset 帧（H.264 压缩噪声使冻结检测无效，此值来自手工确认）
# 帧101显示 t=04.07S → onset ≈ 101-41 = 60-61；手工确认为帧61
ONSET_FRAME_OVERRIDE = 61   # 设为 None 则自动检测（不推荐）

OUT_DIR         = "/tmp"

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def get_digit_roi(gray, timer_roi, digit_subx, digit_suby):
    tx, ty, tw, th = timer_roi
    roi = gray[ty:ty+th, tx:tx+tw]
    return roi[digit_suby, digit_subx].astype(np.float32)


def compute_background(cap, start_frame: int, n_frames: int):
    frames = []
    for fi in range(start_frame, start_frame + n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0)


def preprocess(frame_bgr, bg):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if bg is not None:
        gray = np.clip(gray - bg, 0, 255)
    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def find_blob(processed_gray, y_center: float, search_band: int,
              x_hint=None, x_range=200, x_max=X_MAX):
    """在 y_center±search_band 带内找最亮 blob，返回 (x,y) 或 None。"""
    h, w = processed_gray.shape
    y1 = max(0, int(y_center - search_band))
    y2 = min(h, int(y_center + search_band + 1))
    x1 = max(0, int(x_hint - x_range)) if x_hint is not None else 0
    x2 = min(w, int(x_hint + x_range + 1)) if x_hint is not None else w
    x2 = min(x2, x_max)

    band = processed_gray[y1:y2, x1:x2]
    if band.size == 0:
        return None

    _, max_val, _, max_loc = cv2.minMaxLoc(band)
    if float(max_val) < MIN_BLOB_VAL:
        return None

    bx, by = max_loc
    # 加权质心
    wx = min(10, band.shape[1] // 2)
    wy = min(10, band.shape[0] // 2)
    cx1, cx2 = max(0, bx - wx), min(band.shape[1], bx + wx + 1)
    cy1, cy2 = max(0, by - wy), min(band.shape[0], by + wy + 1)
    patch = band[cy1:cy2, cx1:cx2].astype(np.float64)
    total = patch.sum()
    if total < 1.0:
        return float(x1 + bx), float(y1 + by)
    ys_l, xs_l = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    sub_x = float(np.sum(xs_l * patch) / total) + cx1
    sub_y = float(np.sum(ys_l * patch) / total) + cy1
    return float(x1 + sub_x), float(y1 + sub_y)


def detect_timer_onset(cap, timer_roi=TIMER_ROI, scan_frames=SCAN_FRAMES):
    """
    检测计时器"冻结-解冻"事件（计时器归零重置帧）。
    策略：找到连续 FREEZE_WIN 帧差分 < FREEZE_THR_LOW 之后，
          第一帧差分 > FREEZE_THR_HIGH 的帧即为归零帧。
    """
    diffs = []
    prev_digit = None
    for fi in range(scan_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        digit = get_digit_roi(gray, timer_roi, DIGIT_SUBX, DIGIT_SUBY)
        if prev_digit is not None:
            d = float(np.mean(np.abs(digit - prev_digit)))
            diffs.append((fi, d))
        prev_digit = digit

    if not diffs:
        return -1

    print("  帧间差分（数字ROI）前120帧（每10帧显示一条）：")
    for fi, d in diffs[:120]:
        marker = " <<<" if d > FREEZE_THR_HIGH else ""
        if fi % 10 == 0 or d > FREEZE_THR_HIGH:
            print(f"    fi={fi:4d}  diff={d:.3f}{marker}")

    # 找冻结-解冻事件
    for i in range(FREEZE_WIN, len(diffs)):
        fi_cur, d_cur = diffs[i]
        frozen_period = all(diffs[i - j][1] < FREEZE_THR_LOW
                            for j in range(1, FREEZE_WIN + 1))
        if frozen_period and d_cur > FREEZE_THR_HIGH:
            print(f"\n  ★ 冻结-解冻检测：帧 {fi_cur}（前{FREEZE_WIN}帧冻结，此帧diff={d_cur:.3f}）")
            return fi_cur

    # 退而求其次：前120帧中差分最大的帧
    fi_max = max(diffs[:120], key=lambda x: x[1])[0]
    print(f"  未找到明确冻结-解冻事件，使用前120帧最大差分帧 {fi_max}")
    return fi_max


def annotate_and_save(frame_bgr, particle_pos, y_line_px, label,
                      frame_idx, save_path, particle_color=(0, 255, 0)):
    out = frame_bgr.copy()
    # 画刻度线（黄色）
    cv2.line(out, (0, int(ZERO_LINE_Y)), (frame_bgr.shape[1], int(ZERO_LINE_Y)),
             (0, 220, 220), 1)
    cv2.line(out, (0, int(LINE_16_Y)), (frame_bgr.shape[1], int(LINE_16_Y)),
             (0, 220, 220), 1)
    # 高亮目标刻度线（白色粗线）
    cv2.line(out, (0, int(y_line_px)), (frame_bgr.shape[1], int(y_line_px)),
             (255, 255, 255), 2)
    # 标注粒子位置
    if particle_pos is not None:
        px, py = int(particle_pos[0]), int(particle_pos[1])
        cv2.drawMarker(out, (px, py), particle_color, cv2.MARKER_CROSS, 24, 2)
        cv2.circle(out, (px, py), 12, particle_color, 2)
        cv2.putText(out, f"{label} ({px},{py})",
                    (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, particle_color, 2)
    # 帧号 + 时间
    t_s = frame_idx / FPS
    cv2.putText(out, f"frame={frame_idx}  t={t_s:.2f}s",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.imwrite(save_path, out)
    print(f"  -> 已保存: {save_path}")
    return out


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"无法打开视频: {VIDEO}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频共 {total} 帧，FPS={FPS}")

    # 1. 检测计时器归零帧
    if ONSET_FRAME_OVERRIDE is not None:
        onset = ONSET_FRAME_OVERRIDE
        print(f"\n[Step 1] 使用硬编码 onset = 帧 {onset}  (t={onset/FPS:.2f}s)")
        print("  （H.264压缩噪声使冻结检测失效，onset由手工确认：帧61显示 t=00.21S）")
    else:
        print("\n[Step 1] 检测计时器冻结-解冻事件（归零帧）...")
        onset = detect_timer_onset(cap)
        if onset < 0:
            print("  !! 检测失败，使用默认 onset=61")
            onset = 61
        print(f"  onset = 帧 {onset}  (t={onset/FPS:.2f}s)")

    # 2. 计算背景（onset 之前 BG_FRAMES 帧）
    bg_start = max(0, onset - BG_FRAMES)
    print(f"\n[Step 2] 计算背景 (帧 {bg_start}~{onset})...")
    bg = compute_background(cap, bg_start, BG_FRAMES)

    # 3. 在 onset 帧找 0-线粒子
    print(f"\n[Step 3] 在帧 {onset} 的 0-线附近 (y≈{ZERO_LINE_Y}) 找粒子...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, onset)
    ret, frame_onset = cap.read()
    assert ret

    proc_onset = preprocess(frame_onset, bg)

    # 先用 X_HINT ±X_RANGE 搜索
    pos_0 = find_blob(proc_onset, ZERO_LINE_Y, SEARCH_BAND,
                      x_hint=X_HINT, x_range=X_RANGE)
    if pos_0 is None:
        print(f"  X_HINT={X_HINT} 附近未找到，扩大至全行搜索...")
        pos_0 = find_blob(proc_onset, ZERO_LINE_Y, SEARCH_BAND + 20,
                          x_hint=None, x_range=9999)
    if pos_0 is None:
        print("  !! 仍未找到粒子，用 (X_HINT, ZERO_LINE_Y) 代替")
        pos_0 = (float(X_HINT), ZERO_LINE_Y)

    print(f"  0-线粒子位置: ({pos_0[0]:.1f}, {pos_0[1]:.1f})")
    annotate_and_save(frame_onset, pos_0, ZERO_LINE_Y, "0-line",
                      onset, os.path.join(OUT_DIR, f"crossing_0line_f{onset}.png"),
                      particle_color=(0, 255, 0))

    # 保存背景减除 debug 图
    cv2.imwrite(os.path.join(OUT_DIR, f"debug_proc_f{onset}.png"),
                cv2.normalize(proc_onset, None, 0, 255, cv2.NORM_MINMAX))
    print(f"  -> 已保存背景减除图: /tmp/debug_proc_f{onset}.png")

    # ── Step 4: 前向差分法定位粒子越过1.6线的帧 ─────────────────────────────
    # 原理：以 onset 帧为参考，diff = clip(当前帧 - 参考帧, 0, 255)
    # 刻度线在两帧中均存在 → 相减消除；粒子从 y≈54 移动到新位置 → 亮斑出现在新位置
    print(f"\n[Step 4] 前向差分法追踪粒子 (y={ZERO_LINE_Y:.1f}→{LINE_16_Y:.1f} px)...")

    # 参考帧（onset 帧）灰度 + 模糊
    cap.set(cv2.CAP_PROP_POS_FRAMES, onset)
    ret, ref_frame = cap.read()
    ref_gray = cv2.GaussianBlur(
        cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (7, 7), 0)

    # 预期速度（参考值 v_g≈11.43 µm/s，换算到像素）
    V_EXPECTED_PX_PER_S = 11.43 / 2.381   # ≈ 4.80 px/s
    DIFF_BLOB_MIN       = 5                # 前向差分中最小亮度阈值

    crossing_frame = None
    crossing_pos   = None
    detected_positions = []   # (frame, x, y)

    for fi in range(onset + 1, min(total, onset + 500)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        cur_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (7, 7), 0)

        # 前向差分：仅保留正向变化（粒子出现的新位置）
        fwd_diff = np.clip(cur_gray - ref_gray, 0, 255).astype(np.uint8)

        # 预期粒子 y 坐标
        y_expected = ZERO_LINE_Y + V_EXPECTED_PX_PER_S * (fi - onset) / FPS

        # 搜索带宽比正常大（Brownian 扩散约±15px）
        search_half = 35

        pos = find_blob(fwd_diff, y_expected, search_half,
                        x_hint=X_HINT, x_range=X_RANGE, x_max=X_MAX)
        if pos is not None and fwd_diff[int(pos[1]), int(pos[0])] < DIFF_BLOB_MIN:
            pos = None  # 信号太弱，忽略

        if pos is not None:
            detected_positions.append((fi, pos[0], pos[1]))
            if fi % 20 == 0 or pos[1] > LINE_16_Y - 25:
                print(f"  f={fi:4d} t={fi/FPS:.2f}s  pos=({pos[0]:.1f},{pos[1]:.1f})"
                      f"  y_exp={y_expected:.1f}  diff_max={fwd_diff[int(pos[1]),int(pos[0])]:d}")

            if pos[1] >= LINE_16_Y and crossing_frame is None:
                crossing_frame = fi
                crossing_pos   = pos
                print(f"\n  ★ 粒子越过1.6线！帧={fi}  t={fi/FPS:.2f}s  pos=({pos[0]:.1f},{pos[1]:.1f})")
                break
        else:
            if fi % 40 == 0:
                print(f"  f={fi:4d}  未检测到  y_exp={y_expected:.1f}")

    # 若未检测到越过，根据检测历史估计
    if crossing_frame is None and detected_positions:
        # 找到最靠近 LINE_16_Y 的检测点
        closest = min(detected_positions, key=lambda p: abs(p[2] - LINE_16_Y))
        print(f"\n  !! 未检测到明确越过，使用最近检测点：帧={closest[0]}, pos=({closest[1]:.1f},{closest[2]:.1f})")
        crossing_frame = closest[0]
        crossing_pos   = (closest[1], closest[2])

    # ── Step 5: 保存 1.6-线截图 ──────────────────────────────────────────────
    if crossing_frame is None:
        print("\n  !! 未检测到1.6线越过事件")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, crossing_frame)
    ret, frame_cross = cap.read()
    annotate_and_save(frame_cross, crossing_pos, LINE_16_Y, "1.6-line",
                      crossing_frame,
                      os.path.join(OUT_DIR, f"crossing_1.6line_f{crossing_frame}.png"),
                      particle_color=(0, 100, 255))

    # 前后各1-2帧对比（同样用前向差分）
    x_hint_cross = crossing_pos[0] if crossing_pos else X_HINT
    for offset, tag in [(-2, "before2"), (-1, "before1"), (1, "after1")]:
        fi2 = crossing_frame + offset
        if 0 <= fi2 < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi2)
            ret2, f2 = cap.read()
            if ret2:
                cur2 = cv2.GaussianBlur(
                    cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32), (7, 7), 0)
                diff2 = np.clip(cur2 - ref_gray, 0, 255).astype(np.uint8)
                p2 = find_blob(diff2, LINE_16_Y, SEARCH_BAND + 15,
                               x_hint=x_hint_cross, x_range=X_RANGE + 30)
                annotate_and_save(f2, p2, LINE_16_Y, f"1.6-{tag}", fi2,
                                  os.path.join(OUT_DIR,
                                               f"crossing_1.6_{tag}_f{fi2}.png"),
                                  particle_color=(50, 150, 255))

    dt = (crossing_frame - onset) / FPS
    vg_um_s = 320.0 / dt
    print(f"\n{'='*50}")
    print(f"  onset 帧       : {onset:4d}  (t={onset/FPS:.2f}s)")
    print(f"  1.6-线越过帧   : {crossing_frame:4d}  (t={crossing_frame/FPS:.2f}s)")
    print(f"  Δt             : {dt:.3f} s")
    print(f"  v_g (320µm/Δt) : {vg_um_s:.3f} µm/s")
    print(f"  （参考值: 11.43 µm/s，对应 Δt≈28.0s）")
    print(f"{'='*50}")

    cap.release()
    print("\n完成。图片已保存到 /tmp/")
    print(f"  0-线截图  : /tmp/crossing_0line_f{onset}.png")
    print(f"  1.6-线截图: /tmp/crossing_1.6line_f{crossing_frame}.png")


if __name__ == "__main__":
    main()
