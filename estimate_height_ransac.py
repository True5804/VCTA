# -*- coding: utf-8 -*-
r"""
Estimate camera vertical height h (meters) from traffic frames using only:
- Cars' known height (default 1.5 m; width/length unused for solve)
- Two orthogonal vanishing points (road-direction & vertical)
- YOLO detections + tracking to get many consistent car boxes
- Robust aggregation across cars/frames
- Tiny focal-length micro-adjustment to reduce internal inconsistency (OPTIONAL)

Also outputs:
- Camera pitch (downwards positive, in degrees)
- Ground horizontal range from image bottom (center & min along bottom)

Usage (example):
  python -u estimate_height_ransac.py ^
    --glob "C:\Users\USER\Desktop\accident\T17_10s_frames\*.jpg" ^
    --yolo "yolov8s.pt" --classes 2 7 5 --conf 0.30 ^
    --ultra-track bytetrack --save-vis --vis-dir "track_vis"
"""

import argparse, glob, os, sys, math, json, random
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

Point = Tuple[int, int]
Line  = Tuple[int, int, int, int]  # (x1,y1,x2,y2)

# ============ utilities ============#
def info(msg): print(f"[INFO] {msg}", flush=True)
def oops(msg): print(f"[ERROR] {msg}", flush=True)

# ============ line / VP ============#
def line_from_points(p1: Point, p2: Point) -> np.ndarray:
    P1 = np.array([p1[0], p1[1], 1.0], float)
    P2 = np.array([p2[0], p2[1], 1.0], float)
    l = np.cross(P1, P2)
    n = math.hypot(l[0], l[1])
    if n > 1e-12:
        l /= n
    return l  # ax+by+c=0, sqrt(a^2+b^2)=1

def detect_lines(img: np.ndarray) -> List[Line]:
    """
    貪心版線段偵測：
    - 只看畫面下半部（道路區域），避免天空/建物
    - 降低 Canny / Hough 門檻，讓短一點的線也被抓到
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    # 只取下 70% 當 ROI
    y0 = int(H * 0.3)
    roi = gray[y0:, :]

    # 對比增強
    roi_eq = cv2.equalizeHist(roi)

    lines: List[Line] = []

    # ---------- 1) 先試 FastLineDetector（如果有 ximgproc） ----------#
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector"):
        fld = cv2.ximgproc.createFastLineDetector(
            length_threshold=5,
            canny_th1=20,
            canny_th2=80,
            canny_aperture_size=3,
            do_merge=True,
        )
        segs = fld.detect(roi_eq)
        if segs is not None:
            for s in segs:
                x1, y1, x2, y2 = map(int, s[0])
                y1 += y0
                y2 += y0
                if (x1, y1) != (x2, y2):
                    lines.append((x1, y1, x2, y2))

        if len(lines) >= 8:
            return lines

    # ---------- 2) 再用 HoughLinesP 補強 ----------#
    blur = cv2.GaussianBlur(roi_eq, (3, 3), 0.8)
    edges = cv2.Canny(blur, 30, 90, apertureSize=3, L2gradient=True)

    L = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=int(0.2 * max(H, W)),
        maxLineGap=30,
    )
    if L is not None:
        for x1, y1, x2, y2 in L[:, 0, :]:
            y1 += y0
            y2 += y0
            if (x1, y1) != (x2, y2):
                lines.append((int(x1), int(y1), int(x2), int(y2)))

    return lines


def line_angle(l: Line) -> float:
    x1,y1,x2,y2 = l
    ang = math.atan2((y2-y1), (x2-x1))
    if ang < 0: ang += math.pi
    return ang  # [0,pi)

def pick_axis_groups(lines: List[Line], v_tol_deg=20, min_v=6) -> Tuple[List[Line], List[Line]]:
    """Return (vertical_group, road_group)."""
    if not lines: return [], []
    angs = np.array([line_angle(l) for l in lines])
    best_idx = np.where(np.abs(angs - (np.pi/2)) <= np.radians(v_tol_deg))[0].tolist()
    for tol in (22,25,28,30):
        if len(best_idx) >= min_v: break
        best_idx = np.where(np.abs(angs - (np.pi/2)) <= np.radians(tol))[0].tolist()
    verticals = [lines[i] for i in best_idx]
    remain_idx = [i for i in range(len(lines)) if i not in set(best_idx)]
    remain = [lines[i] for i in remain_idx]
    remain_angs = angs[remain_idx]
    if len(remain)==0: return verticals, []
    hist, edges = np.histogram(remain_angs, bins=36, range=(0.0, math.pi))
    peak = int(np.argmax(hist))
    lo, hi = edges[peak], edges[peak+1]
    width = (hi-lo)*1.2
    center = (lo+hi)/2
    lo2, hi2 = center-width/2, center+width/2
    roads = [L for L,a in zip(remain, remain_angs) if lo2<=a<=hi2]
    return verticals, roads

def vp_from_lines_ransac(lines: List[Line],
                         trials: int = 800,
                         inlier_thresh_px: float = 5.5,
                         min_inliers: int = 6) -> Tuple[Optional[np.ndarray], List[int]]:
    """使用 RANSAC 估計消失點（加上固定 seed，讓兩台機器結果一致）"""
    if len(lines) < 2:
        return None, []

    def line_to_abc(l: Line):
        x1, y1, x2, y2 = l
        a, b, c = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        n = math.hypot(a, b)
        if n > 1e-9:
            a, b, c = a / n, b / n, c / n
        return a, b, c

    L = [line_to_abc(l) for l in lines]
    best_vp = None
    best_inliers = []

    rng = random.Random(0)  # 固定亂數種子

    idx_list = list(range(len(L)))

    for _ in range(trials):
        i, j = rng.sample(idx_list, 2)
        a1, b1, c1 = L[i]
        a2, b2, c2 = L[j]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            continue
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det

        inliers = []
        for k, (a, b, c) in enumerate(L):
            d = abs(a * x + b * y + c)
            if d <= inlier_thresh_px:
                inliers.append(k)

        if len(inliers) > len(best_inliers):
            best_vp = np.array([x, y, 1.0], float)
            best_inliers = inliers

    # 用內點做一次最小平方重算
    if len(best_inliers) >= max(min_inliers, 2):
        A = []
        b = []
        for k in best_inliers:
            a, b_, c = L[k]
            A.append([a, b_])
            b.append([-c])
        A = np.array(A, float)
        b = np.array(b, float)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        best_vp = np.array([x[0, 0], x[1, 0], 1.0], float)
        return best_vp, best_inliers

    return None, []

def vp_from_lines_ls_safe(lines: List[Line]) -> Optional[np.ndarray]:
    """後備：不用 RANSAC，直接最小平方估計 VP。"""
    if len(lines) < 2:
        return None
    A = []
    b = []
    for (x1, y1, x2, y2) in lines:
        l = line_from_points((x1, y1), (x2, y2))  # ax+by+c=0
        A.append([l[0], l[1]])
        b.append([-l[2]])
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([float(x[0][0]), float(x[1][0]), 1.0], float)

def focal_from_orthogonal_vps(vu: Optional[np.ndarray],
                              vv: Optional[np.ndarray],
                              c: Tuple[float,float]) -> float:
    cx,cy = c
    if vu is None or vv is None:
        raise ValueError("vanishing point is None in focal_from_orthogonal_vps")
    du = np.array([vu[0]-cx, vu[1]-cy], float)
    dv = np.array([vv[0]-cx, vv[1]-cy], float)
    f2 = -float(np.dot(du,dv))
    if f2 <= 1e-6:
        f2 = abs(f2) + 1.0
    return math.sqrt(f2)

def dir_from_vp(v: np.ndarray, K: np.ndarray) -> np.ndarray:
    vh = np.array([[v[0]],[v[1]],[1.0]], float)
    d  = np.linalg.inv(K) @ vh
    d  = d[:,0]
    return d/np.linalg.norm(d)

# ============ camera geometry ============#
def solve_KR_from_two_vps(vp_road: np.ndarray, vp_vert: np.ndarray, cx: float, cy: float):
    if vp_road is None or vp_vert is None:
        raise ValueError("vp_road or vp_vert is None in solve_KR_from_two_vps")
    f = focal_from_orthogonal_vps(vp_road, vp_vert, (cx,cy))
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1.0]], float)
    x_dir = dir_from_vp(vp_road, K)  # world x (along road)
    y_dir = dir_from_vp(vp_vert, K)  # world y (up)
    z_dir = np.cross(x_dir, y_dir); z_dir /= np.linalg.norm(z_dir)
    x_dir = np.cross(y_dir, z_dir); x_dir /= np.linalg.norm(x_dir)
    R = np.column_stack([x_dir, y_dir, z_dir])  # world->camera
    return K, R, f

def ground_horizon_line(K: np.ndarray, R: np.ndarray) -> np.ndarray:
    n_world = np.array([0.0,1.0,0.0], float)
    n_cam   = R @ n_world
    l = np.linalg.inv(K).T @ n_cam  # a,b,c
    ab = math.hypot(l[0], l[1])
    if ab>1e-12: l /= ab
    return l

def horizon_y(l: np.ndarray, x: float) -> float:
    a,b,c = l
    if abs(b)<1e-12: return float('inf')
    return -(a*x + c)/b

def ray_dir_from_pixel(u: float, v: float, K: np.ndarray) -> np.ndarray:
    d = np.linalg.inv(K) @ np.array([[u],[v],[1.0]], float)
    d = d[:,0]
    return d/np.linalg.norm(d)

def height_from_box(u_c, v_t, v_b, K, R, H=1.5) -> Optional[float]:
    """Solve h from one car box using vertical constraint."""
    r_b = ray_dir_from_pixel(u_c, v_b, K)
    r_t = ray_dir_from_pixel(u_c, v_t, K)
    d_b = R.T @ r_b
    d_t = R.T @ r_t
    denx = (-d_b[0]/d_b[1] + d_t[0]/d_t[1])
    hx = (H * (d_t[0]/d_t[1]))/denx if abs(denx)>1e-9 else None
    denz = (-d_b[2]/d_b[1] + d_t[2]/d_t[1])
    hz = (H * (d_t[2]/d_t[1]))/denz if abs(denz)>1e-9 else None
    cand=[h for h in (hx,hz) if h is not None and np.isfinite(h) and 0.5<h<50.0]
    if not cand: return None
    if len(cand)==2 and abs(cand[0]-cand[1])>3.0:  # inconsistent
        return None
    return float(np.mean(cand))

def camera_pitch_deg(R):
    """Downwards-positive pitch in degrees, using ry = R[:,1]."""
    ry = R[:, 1]
    z = float(ry[2])
    z = max(-1.0, min(1.0, z))
    return math.degrees(math.asin(z))

def ground_range_for_pixel(u, v, K, ry, height_m):
    """
    Intersect pixel ray with ground plane. Auto-flip normal if needed so that
    n^T d < 0, then t = -h / (n^T d); horizontal range = sqrt(t^2 - h^2).
    """
    d = ray_dir_from_pixel(u, v, K)
    denom = float(np.dot(ry, d))
    if denom >= -1e-9:
        denom2 = float(np.dot(-ry, d))
        if denom2 >= -1e-9:
            return None
        ry = -ry
        denom = denom2
    t = -height_m / denom
    if t <= 0:
        return None
    return math.sqrt(max(0.0, t*t - height_m*height_m))

# ============ YOLO / tracking ============#
def run_ultra_tracker(frames, model_path, classes, conf=0.30, tracker_name="bytetrack", device="cpu"):
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="ultra_track_")
    try:
        for i,im in enumerate(frames):
            cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), im)
        model = YOLO(model_path)
        tracker_yaml = "botsort.yaml" if tracker_name.lower()=="botsort" else "bytetrack.yaml"
        # 嘗試用指定裝置；若 CUDA 失敗，自動退回 CPU
        try:
            results = model.track(
                source=tmpdir, conf=conf, classes=list(classes),
                tracker=tracker_yaml, stream=True, verbose=False,
                save=False, persist=True, device=device
            )
        except Exception as e:
            msg = str(e)
            warn_keywords = ("invalid device id", "no kernel image", "cuda", "cudnn")
            if any(k in msg.lower() for k in warn_keywords):
                #info("Device init failed, fallback to CPU.")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                results = model.track(
                    source=tmpdir, conf=conf, classes=list(classes),
                    tracker=tracker_yaml, stream=True, verbose=False,
                    save=False, persist=True, device="cpu"
                )
            else:
                raise
        out=[[] for _ in range(len(frames))]

        i=-1
        for r in results:
            i+=1
            if r.boxes is None: continue
            ids = r.boxes.id
            if ids is None:
                for b,c,s in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy()):
                    x1,y1,x2,y2 = b.astype(int)
                    out[i].append([x1,y1,x2,y2,int(c),float(s), -1])
            else:
                ids = ids.int().cpu().numpy()
                for b,c,s,tid in zip(r.boxes.xyxy.cpu().numpy(),
                                     r.boxes.cls.cpu().numpy(),
                                     r.boxes.conf.cpu().numpy(),
                                     ids):
                    x1,y1,x2,y2 = b.astype(int)
                    out[i].append([x1,y1,x2,y2,int(c),float(s), int(tid)])
        return out
    finally:
        import shutil as _sh
        _sh.rmtree(tmpdir, ignore_errors=True)

# ============ filters & helpers ============#
def intersect_line_with_segment(line_ab: np.ndarray, seg: Tuple[Point,Point]) -> Optional[Tuple[float,float]]:
    (x1,y1),(x2,y2) = seg
    l1 = np.cross(np.array([x1,y1,1.0]), np.array([x2,y2,1.0]))
    p = np.cross(line_ab, l1)
    if abs(p[2])<1e-12: return None
    x,y = p[0]/p[2], p[1]/p[2]
    xmin, xmax = sorted([x1,x2]); ymin,ymax = sorted([y1,y2])
    tol = 1e-3
    if (xmin-tol)<=x<=(xmax+tol) and (ymin-tol)<=y<=(ymax+tol):
        return (float(x), float(y))
    return None

def horizon_filter(box, l_hor, W, H, min_box=12, margin_px=6, max_area_frac=0.25):
    x1,y1,x2,y2,cls,conf,tid = box
    w=x2-x1; h=y2-y1
    if w<min_box or h<min_box: return False
    if x1<=1 or y1<=1 or x2>=W-1 or y2>=H-1: return False
    if (w*h) > max_area_frac*W*H: return False
    ar = w/max(h,1e-6)
    if not (0.4<=ar<=4.5): return False
    u_c = (x1+x2)/2.0; v_b = y2
    y_h = horizon_y(l_hor, u_c)
    if not np.isfinite(y_h):
        return False
    # 只有在「地平線落在畫面附近」時，才要求車子在地平線下方
    if -margin_px <= y_h <= H + margin_px:
        if v_b <= y_h + margin_px:
            return False
    # 若地平線遠遠在畫面下方，就不要用這個條件砍掉車子
    return True

def median_and_mad(arr: np.ndarray):
    if arr.size==0: return None, None
    med = np.median(arr)
    mad = np.median(np.abs(arr-med))+1e-6
    return float(med), float(mad)

def micro_adjust_f(K_init, R_init, vp_road, vp_vert, cx, cy, samples, H_car, h_vals_solver):
    f0 = K_init[0,0]
    best = (float('inf'), f0)  # (mad, f)
    for s in samples:
        f = f0 * s
        K = np.array([[f,0,cx],[0,f,cy],[0,0,1.0]], float)
        x_dir = dir_from_vp(vp_road, K)
        y_dir = dir_from_vp(vp_vert, K)
        z_dir = np.cross(x_dir, y_dir); z_dir/=np.linalg.norm(z_dir)
        x_dir = np.cross(y_dir, z_dir); x_dir/=np.linalg.norm(x_dir)
        R = np.column_stack([x_dir, y_dir, z_dir])
        hs = h_vals_solver(K,R)
        hs = np.array([h for h in hs if h is not None and np.isfinite(h) and 0.5<h<50.0], float)
        if hs.size<6: continue
        _, mad = median_and_mad(hs)
        if mad is not None and mad < best[0]:
            best = (mad, f)
    return best[1]

# ============ main ============#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*", help="Image list (ordered)")
    ap.add_argument("--glob", type=str, default=None, help="Glob, e.g. 'frame_*.jpg'")
    ap.add_argument("--yolo", type=str, default="yolov8s.pt", help="Ultralytics weights")
    ap.add_argument("--classes", type=int, nargs="*", default=[2,7,5], help="COCO ids (car=2 truck=7 bus=5)")
    ap.add_argument("--conf", type=float, default=0.30, help="YOLO confidence")
    ap.add_argument("--carH", type=float, default=1.5, help="Car height prior H (m)")
    ap.add_argument("--ultra-track", type=str, choices=["bytetrack","botsort"], default="bytetrack",
                    help="Use Ultralytics tracker")
    ap.add_argument("--device", type=str, default="cpu", help="torch device: 'cpu', '0', '0,1'")
    ap.add_argument("--min-box", type=int, default=12, help="Ignore too small boxes")
    ap.add_argument("--save-vis", action="store_true", help="Save visualization frames")
    ap.add_argument("--vis-dir", type=str, default="track_vis", help="Vis folder")
    ap.add_argument("--no-micro-adjust", action="store_true", help="Disable focal micro adjustment")
    args = ap.parse_args()

    # load frames
    paths=[]
    if args.glob:
        paths = sorted(glob.glob(args.glob))
    if args.frames:
        paths += list(args.frames)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        oops("No images found. Check --glob or --frames."); sys.exit(1)

    frames=[]
    for p in paths:
        im = cv2.imread(p)
        if im is not None: frames.append(im)
    if not frames:
        oops("Failed to read any image."); sys.exit(1)

    # info(f"Read {len(frames)} frames.")
    H_img, W_img = frames[0].shape[:2]
    cx, cy = W_img/2.0, H_img/2.0

    # ---------- VP from first clear frame ----------#
    # info("Detecting lines & vanishing points ...")

    # 垂直線 / 道路線 最小門檻
    MIN_VERT_LINES = 6
    MIN_ROAD_LINES = 10

    anchor = None
    lines = None
    L_vert = None
    L_road = None

    for idx, frame in enumerate(frames):
        cand_lines = detect_lines(frame)
        #info(f"Frame {idx}: detected {len(cand_lines)} line segments.")
        if len(cand_lines) < 8:
            continue

        Lv, Lr = pick_axis_groups(cand_lines, v_tol_deg=30, min_v=3)

        if len(Lv) < 3 or len(Lr) < 2:
            Lv2, Lr2 = pick_axis_groups(cand_lines, v_tol_deg=25, min_v=4)
            if len(Lv2) + len(Lr2) > len(Lv) + len(Lr):
                Lv, Lr = Lv2, Lr2

        #info(f"Frame {idx}: vertical lines = {len(Lv)}, road lines = {len(Lr)}")

        if len(Lv) < MIN_VERT_LINES or len(Lr) < MIN_ROAD_LINES:
            #info(f"Frame {idx}: not enough vertical/road lines, skip.")
            continue

        anchor = frame
        lines = cand_lines
        L_vert = Lv
        L_road = Lr
        #info(f"👉 Using frame {idx} as VP anchor (vert={len(L_vert)}, road={len(L_road)}).")
        break

    if anchor is None or lines is None or L_vert is None or L_road is None:
        oops("Could not find any frame with robust vertical/road line groups."); sys.exit(1)

    # ---------- 先用 RANSAC 估 VP，失敗就用 LS fallback ----------#
    vp_road, _ = vp_from_lines_ransac(L_road, trials=1500, inlier_thresh_px=3.0)
    vp_vert, _ = vp_from_lines_ransac(L_vert, trials=1500, inlier_thresh_px=3.0)

    if vp_road is None or vp_vert is None:
        #info("[WARN] RANSAC 無法穩定估計某一個 vanishing point，改用 least-squares 後備方案。")
        if vp_road is None:
            vp_road = vp_from_lines_ls_safe(L_road)
        if vp_vert is None:
            vp_vert = vp_from_lines_ls_safe(L_vert)

    if vp_road is None or vp_vert is None:
        oops("無法從線段估計 vanishing points（RANSAC + LS 都失敗），請檢查畫面是否有足夠道路線/垂直線。")
        sys.exit(1)

    # try both assignments & pick plausible
    cand=[]
    for vpR,vpV in ((vp_road,vp_vert),(vp_vert,vp_road)):
        try:
            K_tmp, R_tmp, f_tmp = solve_KR_from_two_vps(vpR, vpV, cx, cy)
        except ValueError as e:
            #info(f"[WARN] solve_KR_from_two_vps 失敗：{e}")
            continue

        y_dir = R_tmp[:,1]
        l_tmp = ground_horizon_line(K_tmp, R_tmp)
        y_h_center = horizon_y(l_tmp, cx)

        score = 0.0

        # 垂直 VP 要在影像中心上方：越高越好（加分），在下方就扣很大
        if vpV[1] < cy:
            score += (cy - vpV[1]) * 2.0
        else:
            score -= (vpV[1] - cy) * 5.0

        # 若地平線跑到畫面底下很遠，直接大扣分（這就是你剛剛遇到的情況）
        if y_h_center > (H_img + 10):
            score -= 1e5

        # world-up 在 camera 座標中，y 分量應該是負的（往上）
        if y_dir[1] > -0.1:
            score -= 2000.0

        # 水平偏斜太多也扣分
        score -= 500.0 * abs(y_dir[0])

        cand.append((score, (vpR, vpV, K_tmp, R_tmp, f_tmp)))

    if not cand:
        oops("所有 VP 指派組合都不合理，請檢查線段偵測與 vanishing point 分群。")
        sys.exit(1)

    cand.sort(key=lambda x:x[0], reverse=True)
    vpR, vpV, K, R, f = cand[0][1]
    l_hor = ground_horizon_line(K,R)
    #info(f"Initial focal f≈{f:.2f}px, horizon line=[{l_hor[0]:.5f},{l_hor[1]:.5f},{l_hor[2]:.2f}]")

    # ---------- YOLO + tracking ----------#
    if YOLO is None:
        oops("Ultralytics not installed. pip install ultralytics"); sys.exit(1)
    #info(f"Tracking with Ultralytics: {args.ultra_track}")

    if args.device.strip().lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    dets_by_frame = run_ultra_tracker(frames, model_path=args.yolo,
                                      classes=tuple(args.classes), conf=args.conf,
                                      tracker_name=args.ultra_track, device=args.device)

    # ---------- collect candidate boxes ----------#
    tracks = defaultdict(list)  # tid -> list of (fidx, [x1,y1,x2,y2], conf, cls)
    assigned = 0
    for fi, dets in enumerate(dets_by_frame):
        keep=[]
        for x1,y1,x2,y2,cls,conf,tid in dets:
            if tid<0:
                continue
            if not horizon_filter([x1,y1,x2,y2,cls,conf,tid], l_hor, W_img, H_img,
                                  min_box=args.min_box, margin_px=6, max_area_frac=0.30):
                continue
            keep.append([x1,y1,x2,y2,cls,conf,tid])
        for x1,y1,x2,y2,cls,conf,tid in keep:
            tracks[tid].append((fi, [x1,y1,x2,y2], float(conf), int(cls)))
        assigned += len(keep)
        if args.save_vis:
            vis = frames[fi].copy()
            for x1, y1, x2, y2, cls, conf, tid in keep:
                color = (0,255,0) if cls==2 else (0,165,255) if cls==7 else (0,0,255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"ID {tid} {('car' if cls==2 else 'truck' if cls==7 else 'bus')} {conf:.2f}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ytxt = max(0, y1 - 4)
                cv2.rectangle(vis, (x1, ytxt - th - 4), (x1 + tw + 4, ytxt + bl), color, -1)
                cv2.putText(vis, label, (x1 + 2, ytxt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            os.makedirs(args.vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.vis_dir, f"det_{fi:03d}.jpg"), vis)

    #info(f"Kept {assigned} filtered boxes across {len(tracks)} track IDs.")

    # ---------- per-box height solve ----------#
    def top_point_along_vertical(u_b, v_b, box, vp_vert) -> Optional[Tuple[float,float]]:
        x1,y1,x2,y2, *_ = box
        L = line_from_points((float(u_b), float(v_b)), (vp_vert[0], vp_vert[1]))
        inter = intersect_line_with_segment(L, ((x1,y1),(x2,y1)))
        if inter is None:
            return ((x1+x2)/2.0, float(y1))
        return inter

    def solve_all_heights(K_use, R_use):
        h_vals=[]
        for tid, hist in tracks.items():
            for (fi, bb, conf, cls) in hist:
                x1,y1,x2,y2 = bb
                u_b = (x1+x2)/2.0; v_b = float(y2)
                tp = top_point_along_vertical(u_b, v_b, [x1,y1,x2,y2], vpV)
                if tp is None:
                    continue
                u_t, v_t = tp
                h_i = height_from_box(u_c=u_b, v_t=v_t, v_b=v_b, K=K_use, R=R_use, H=args.carH)
                h_vals.append((tid, fi, h_i))
        return h_vals

    raw_h = solve_all_heights(K, R)
    hs = np.array([h for (_,_,h) in raw_h if h is not None and np.isfinite(h)], float)
    if hs.size==0:
        oops("No valid per-box height could be solved. Try lowering --conf or check frames."); sys.exit(2)

    med0, mad0 = median_and_mad(hs)
    inlier = np.abs(hs - med0) < 2.5*mad0
    hs1 = hs[inlier]
    #info(f"Raw h count={hs.size}, inliers={hs1.size}, median≈{np.median(hs1):.2f} m, MAD≈{np.median(np.abs(hs1-np.median(hs1))):.2f}")

    # ---------- micro-adjust f (optional, tiny) ----------#
    if not args.no_micro_adjust:
        # 以原本 f 為中心，±15% 小範圍掃描
        scales = [1.0 + s for s in np.linspace(-0.15, 0.15, 13)]

        def solver_with(Ktmp, Rtmp):
            hs_tmp = []
            for tid, hist in tracks.items():
                for (fi, bb, conf, cls) in hist:
                    x1, y1, x2, y2 = bb
                    u_b = (x1 + x2) / 2.0
                    v_b = float(y2)
                    tp = top_point_along_vertical(u_b, v_b, [x1, y1, x2, y2], vpV)
                    if tp is None:
                        continue
                    u_t, v_t = tp
                    h_i = height_from_box(
                        u_c=u_b, v_t=v_t, v_b=v_b,
                        K=Ktmp, R=Rtmp, H=args.carH
                    )
                    hs_tmp.append(h_i)
            return hs_tmp

        f_best = micro_adjust_f(K, R, vpR, vpV, cx, cy, scales, args.carH, solver_with)
        if abs(f_best - f) > 1e-6:
            f = f_best
            # 用最佳 f 重算 K, R, horizon
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], float)
            x_dir = dir_from_vp(vpR, K)
            y_dir = dir_from_vp(vpV, K)
            z_dir = np.cross(x_dir, y_dir)
            z_dir /= np.linalg.norm(z_dir)
            x_dir = np.cross(y_dir, z_dir)
            x_dir /= np.linalg.norm(x_dir)
            R = np.column_stack([x_dir, y_dir, z_dir])
            l_hor = ground_horizon_line(K, R)
            #info(f"Micro-adjusted focal f≈{f:.2f}px; recomputing heights...")

            raw_h = solve_all_heights(K, R)
            hs = np.array(
                [h for (_, _, h) in raw_h if h is not None and np.isfinite(h)],
                float
            )
            if hs.size >= 6:
                med0, mad0 = median_and_mad(hs)
                inlier = np.abs(hs - med0) < 2.5 * mad0
                hs1 = hs[inlier]

    # ---------- aggregate per-track, then globally ----------#
    per_track = defaultdict(list)
    for (tid, fi, h_i) in raw_h:
        if h_i is not None and np.isfinite(h_i):
            per_track[tid].append(h_i)

    track_vals = []
    for tid, arr in per_track.items():
        arr = np.array(arr, float)
        if arr.size < 3:
            continue
        med_t, mad_t = median_and_mad(arr)
        if mad_t is None or mad_t > 2.5:
            continue
        track_vals.append((tid, med_t, mad_t, len(arr)))

    if not track_vals:
        oops("No stable tracks after filtering."); sys.exit(3)

    weights = []
    values = []
    for tid, med_t, mad_t, L in track_vals:
        # 觀測多、MAD 小者權重大
        w = max(1.0, L) / (1.0 + max(0.1, mad_t))
        weights.append(w)
        values.append(med_t)
    weights = np.array(weights, float)
    values = np.array(values, float)
    h_est = float(np.average(values, weights=weights))

    # ---------- pitch & ground ranges ----------#
    pitch_deg = camera_pitch_deg(R)
    ry = R[:, 1]

    yb = H_img - 1
    u_center = (W_img - 1) / 2.0

    R_center = ground_range_for_pixel(u_center, yb, K, ry, h_est)
    xs = np.linspace(0, W_img - 1, num=min(100, W_img))
    R_candidates = []
    for u in xs:
        Ru = ground_range_for_pixel(u, yb, K, ry, h_est)
        if Ru is not None and np.isfinite(Ru):
            R_candidates.append(Ru)
    R_min = float(min(R_candidates)) if R_candidates else None

    # ---------- report ----------#
    print("\n===== Camera Height Estimation =====", flush=True)
    # print(f"Frames: {len(frames)}  | Image: {W_img}x{H_img}", flush=True)
    print(f"VP road: ({vpR[0]:.1f},{vpR[1]:.1f})  VP vert: ({vpV[0]:.1f},{vpV[1]:.1f})", flush=True)
    # print(f"Focal f ≈ {f:.2f} px", flush=True)
    print(f"Horizon line: [{l_hor[0]:.5f},{l_hor[1]:.5f},{l_hor[2]:.2f}]", flush=True)
    # print(f"Per-box valid: {hs.size}  Inliers used: {hs1.size}", flush=True)
    # print(f"Per-track used: {len(track_vals)}", flush=True)
    print(f"\n✅ Estimated camera height h ≈ {h_est:.2f} m  (car height H={args.carH:.2f} m)", flush=True)
    print(f"Pitch (down +): {pitch_deg:.2f} deg", flush=True)
    # if R_center is not None:
        # print(f"Bottom-center ground range ≈ {R_center:.2f} m", flush=True)
    # if R_min is not None:
        # print(f"Bottom-line min ground range ≈ {R_min:.2f} m", flush=True)

    # ---------- visualization of VPs & horizon ----------#
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        vis = anchor.copy()

        def draw_lines(vimg, L, color):
            for (x1, y1, x2, y2) in L:
                cv2.line(vimg, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        draw_lines(vis, L_road, (0, 255, 0))
        draw_lines(vis, L_vert, (0, 128, 255))

        def cross(img, p, col):
            x, y = int(p[0]), int(p[1])
            s = 10
            cv2.line(img, (x - s, y), (x + s, y), col, 2, cv2.LINE_AA)
            cv2.line(img, (x, y - s), (x, y + s), col, 2, cv2.LINE_AA)

        cross(vis, (vpR[0], vpR[1]), (0, 255, 0))
        cross(vis, (vpV[0], vpV[1]), (0, 128, 255))

        xs = np.linspace(0, W_img - 1, 50)
        pts = []
        for x in xs:
            y = horizon_y(l_hor, x)
            if np.isfinite(y):
                pts.append((int(x), int(round(y))))
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(
            vis, f"f={f:.1f}px  h~{h_est:.2f}m",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        txt2 = f"pitch={pitch_deg:.2f}  Rbc~{(R_center if R_center is not None else float('nan')):.2f}m"
        cv2.putText(
            vis, txt2,
            (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )

        outp = os.path.join(args.vis_dir, "vp_horizon.png")
        cv2.imwrite(outp, vis)
        #info(f"Saved VP/horizon visualization: {outp}")

    # ---------- JSON summary ----------#
    summary = {
        "frames": len(frames),
        "image_size": [int(W_img), int(H_img)],
        "vp_road": [float(vpR[0]), float(vpR[1])],
        "vp_vert": [float(vpV[0]), float(vpV[1])],
        "focal_px": float(f),
        "h_estimate_m": float(h_est),
        "car_height_m": float(args.carH),
        "per_track_count": int(len(track_vals)),
        "per_box_valid": int(hs.size),
        "per_box_inliers": int(hs1.size),
        "pitch_deg_down_positive": float(pitch_deg),
        "ground_range_bottom_center_m": None if R_center is None else float(R_center),
        "ground_range_bottom_min_m": None if R_min is None else float(R_min),
    }
    out_json = os.path.join(args.vis_dir if args.save_vis else ".", "height_est_summary.json")
    with open(out_json, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)
        #info(f"Wrote {out_json}")

if __name__ == "__main__":
    main()

