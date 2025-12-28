# -*- coding: utf-8 -*-
import argparse, os, glob, json, math
import numpy as np, cv2, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 強制關 GPU
from skimage.transform import probabilistic_hough_line

# -------------------- MiDaS --------------------
def load_midas(variant_prefer="MiDaS_small"):
    device = torch.device("cpu")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    for v in [variant_prefer, "DPT_Hybrid", "DPT_Large"]:
        try:
            model = torch.hub.load("intel-isl/MiDaS", v).to(device).eval()
            tfm = transforms.small_transform if "small" in v.lower() else transforms.dpt_transform
            # print(f"[MiDaS] using: {v} on CPU")
            return model, tfm, device
        except Exception as e:
            print(f"[MiDaS] load {v} failed: {e}")
    raise RuntimeError("MiDaS load failed")

@torch.inference_mode()
def midas_infer(model, transform, device, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).to(device)
    pred = model(inp)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    # normalize to positive
    pred = pred - np.min(pred) + 1e-9
    return pred  # larger ≈ nearer

def fuse_depth_median(imgs, model, tfm, device):
    ds = [midas_infer(model, tfm, device, im) for im in imgs]
    D = np.stack(ds, 0)
    med = np.median(D, 0)
    # Normalize by median of positive values
    pos = med[med > 0]
    if pos.size > 0:
        med /= np.median(pos)
    return med.astype(np.float64)

# -------------------- 幾何 / 消失點 --------------------
def line_from_pts(p1,p2):
    return np.cross(np.array([p1[0],p1[1],1.],np.float64),
                    np.array([p2[0],p2[1],1.],np.float64))

def detect_lines(img, canny1=60, canny2=160, min_len=60, gap=6):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(3,3),0)
    edges = cv2.Canny(g, canny1, canny2)
    segs = probabilistic_hough_line(edges, threshold=10,
                                    line_length=min_len, line_gap=gap)
    lines=[]
    for (x1,y1),(x2,y2) in segs:
        if (x1,y1)!=(x2,y2): lines.append(((x1,y1),(x2,y2)))
    return lines

def split_two_families(lines, angle_tol_deg=12):
    road_like=[]; vert_like=[]
    for (x1,y1),(x2,y2) in lines:
        ang = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(ang) < (angle_tol_deg+13):           # near horizontal
            road_like.append(((x1,y1),(x2,y2)))
        if abs(abs(ang)-90) < (angle_tol_deg+3):    # near vertical
            vert_like.append(((x1,y1),(x2,y2)))
    return road_like, vert_like

def intersect_lines_ls(lines):
    A=[]; b=[]
    for (p1,p2) in lines:
        a,b_,c = line_from_pts(p1,p2)
        A.append([a,b_]); b.append([-c])
    A=np.array(A,dtype=np.float64); b=np.array(b,dtype=np.float64)
    x,_,_,_=np.linalg.lstsq(A,b,rcond=None)
    return np.array([x[0,0], x[1,0], 1.0], np.float64)

def focal_from_orthogonal_vps(v1, v2, cx, cy):
    a = np.array([v1[0]-cx, v1[1]-cy], float)
    b = np.array([v2[0]-cx, v2[1]-cy], float)
    f2 = -float(np.dot(a,b))
    return math.sqrt(max(f2, 1e-6))

def dir_from_vp(v, K):
    vh = np.array([[v[0]],[v[1]],[1.0]], float)
    d  = np.linalg.inv(K) @ vh
    d  = d[:,0]
    return d/np.linalg.norm(d)

def solve_KR_from_two_vps(vp_road, vp_vert, cx, cy):
    f = focal_from_orthogonal_vps(vp_road, vp_vert, cx, cy)
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1.0]], float)
    x_dir = dir_from_vp(vp_road, K)
    y_dir = dir_from_vp(vp_vert, K)
    z_dir = np.cross(x_dir, y_dir); z_dir/=np.linalg.norm(z_dir)
    x_dir = np.cross(y_dir, z_dir); x_dir/=np.linalg.norm(x_dir)
    R = np.column_stack([x_dir, y_dir, z_dir])  # world->camera
    return K, R, f

def solve_R_from_two_vps_with_known_K(vp_road, vp_vert, K):
    """已知 K（例如來自 JSON 的 f 與影像中心），僅由兩個 VP 解 R，並做正交化。"""
    x_dir = dir_from_vp(vp_road, K)
    y_dir = dir_from_vp(vp_vert, K)
    z_dir = np.cross(x_dir, y_dir); z_dir /= (np.linalg.norm(z_dir) + 1e-12)
    x_dir = np.cross(y_dir, z_dir); x_dir /= (np.linalg.norm(x_dir) + 1e-12)
    y_dir = np.cross(z_dir, x_dir); y_dir /= (np.linalg.norm(y_dir) + 1e-12)
    R = np.column_stack([x_dir, y_dir, z_dir])  # world->camera
    return R

# --------- 校正 R 的朝向：確保影像底部視線能指向地面 ----------
def orient_R_for_ground(K, R, W, H):
    """用幾個底部像素檢查 r_w[1]，若傾向向上，翻轉 R 的 y 軸並重新正交化。"""
    Ki = np.linalg.inv(K)
    def ray_world(u, v):
        r_cam = Ki @ np.array([u, v, 1.0], float)
        r_cam /= (np.linalg.norm(r_cam) + 1e-12)
        return R.T @ r_cam
    probes = [(int(W*0.5), int(H*0.95)), (int(W*0.5), int(H*0.90)), (int(W*0.5), int(H*0.85))]
    ys = [ray_world(u,v)[1] for (u,v) in probes]
    if np.mean(ys) >= 0.0:
        R = R.copy()
        R[:,1] *= -1.0                         # 翻轉 y 軸
        z = np.cross(R[:,0], R[:,1]); z /= (np.linalg.norm(z) + 1e-12)
        x = np.cross(R[:,1], z); x /= (np.linalg.norm(x) + 1e-12)
        R = np.column_stack([x, R[:,1], z])
    return R

def ground_intersect(u, v, K, R, h, allow_backside=True):
    """
    回傳 (X, Y=0, Z, D, behind_flag)。
    若 allow_backside=True，當正向射線無法落到地面時，嘗試使用反向射線（等價於在相機後方的地面）。
    """
    Ki = np.linalg.inv(K)
    r_cam = Ki @ np.array([u, v, 1.0], np.float64)
    nrm = np.linalg.norm(r_cam) + 1e-12
    r_cam = r_cam / nrm

    # 兩個方向都試：先正向，再反向
    for sgn in (1.0, -1.0) if allow_backside else (1.0,):
        r_w = R.T @ (sgn * r_cam)      # 轉到世界座標
        if r_w[1] < -1e-9:             # 必須往地面（-Y）
            t = -h / r_w[1]
            if t > 0:
                X = t * r_w[0]
                Z = t * r_w[2]
                D = float(np.hypot(X, Z))
                behind = (sgn < 0.0)   # 用了反向射線 → 在相機後方
                return X, 0.0, Z, D, behind

    return None, None, None, None, None

# ------- QC：底部附近第一個有效地面交點 -------
def qc_bottom_center_range(K, R, h, W, H):
    """
    回傳 (D, (u,v), frac, behind)；若找不到則 D=None。
    只接受在鏡頭前方的交點（allow_backside=False）。
    """
    for frac in [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86]:
        u = int(W * 0.5)
        v = int(H * frac)
        X, Y, Z, D, behind = ground_intersect(u, v, K, R, h, allow_backside=False)
        if D is not None:
            return D, (u, v), frac, behind
    return None, (int(W * 0.5), int(H * 0.98)), None, None

# -------------------- 決定性 MiDaS 尺度擬合 --------------------
def fit_midas_scale_stable(depth_rel, K, R, h,
                           grid_step=6,
                           x_range=(0.20, 0.80),
                           y_range=(0.55, 0.98),
                           trim=(10.0, 90.0),
                           dmin=3.0, dmax=120.0,
                           min_samples=20):
    H, W = depth_rel.shape
    xs = np.arange(int(W*x_range[0]), int(W*x_range[1]), grid_step, dtype=int)
    ys = np.arange(int(H*y_range[0]), int(H*y_range[1]), grid_step, dtype=int)
    Ki = np.linalg.inv(K)

    def ray_world(u, v):
        r_cam = Ki @ np.array([u, v, 1.0], np.float64)
        r_cam /= (np.linalg.norm(r_cam) + 1e-12)
        return R.T @ r_cam

    c_all = c_dr = c_down = c_tpos = c_dist = 0
    s_candidates = []
    for y in ys:
        for x in xs:
            c_all += 1
            dr = float(depth_rel[y, x])
            if not np.isfinite(dr) or dr <= 1e-9:  # 深度無效
                continue
            c_dr += 1
            r_w = ray_world(x, y)
            if r_w[1] >= -1e-9:                   # 必須朝下
                continue
            c_down += 1
            t = -h / r_w[1]
            if t <= 0:                             # 必須在前方
                continue
            c_tpos += 1
            X = t * r_w[0]; Z = t * r_w[2]
            Dg = float(np.hypot(X, Z))
            if not (dmin <= Dg <= dmax):           # 幾何距離門檻
                continue
            c_dist += 1
            Rg = math.sqrt(Dg*Dg + h*h)
            s_i = Rg * dr
            if np.isfinite(s_i) and s_i > 0:
                s_candidates.append(s_i)

    if len(s_candidates) < min_samples:
        # print(f"[DEBUG] samples: total={c_all}, dr_ok={c_dr}, down_ok={c_down}, tpos_ok={c_tpos}, dist_ok={c_dist}, s_ok={len(s_candidates)}")
        return None, {"count": len(s_candidates), "note": "too_few_samples"}

    s_arr = np.array(s_candidates, float)
    lo, hi = np.percentile(s_arr, [trim[0], trim[1]])
    inliers = s_arr[(s_arr >= lo) & (s_arr <= hi)]
    if inliers.size == 0:
        #print(f"[DEBUG] percentile trim removed all; lo={lo:.3f}, hi={hi:.3f}, N={len(s_arr)}")
        return None, {"count": len(s_candidates), "note": "all_trimmed"}

    s_med = float(np.median(inliers))
    #print(f"[DEBUG] samples: total={c_all}, dr_ok={c_dr}, down_ok={c_down}, tpos_ok={c_tpos}, dist_ok={c_dist}, s_ok={len(s_candidates)}, inliers={inliers.size}")
    return s_med, {"count": len(s_candidates), "inliers": int(inliers.size), "lo": float(lo), "hi": float(hi)}

# -------------------- 方案 A／錨點校正（three anchors + 增益） --------------------
def _anchor_pixels(kind, W, H):
    v = int(H * 0.98)
    if kind == "three":
        return [(int(W*0.35), v), (int(W*0.50), v), (int(W*0.65), v)]
    return [(int(W*0.50), v)]  # center

def compute_s_with_anchor(depth_rel, K, R, h, s_med,
                          anchor_kind="three",
                          anchor_target_mult=1.15,
                          anchor_gain=1.10,
                          W=None, H=None):
    if s_med is None:
        return None, {"mode": "anchor", "note": "no_s_med"}
    Ki = np.linalg.inv(K)
    def ray_world(u, v):
        r_cam = Ki @ np.array([u, v, 1.0], float)
        r_cam /= (np.linalg.norm(r_cam) + 1e-12)
        return R.T @ r_cam

    anchors = _anchor_pixels(anchor_kind, W, H)
    s_list = []
    dbg = {"anchors": 0, "s_anchor_raw": None, "k": anchor_target_mult, "gain": anchor_gain}
    for (u, v) in anchors:
        r_w = ray_world(u, v)
        if r_w[1] >= -1e-9: continue
        t = -h / r_w[1]
        if t <= 0: continue
        X = t * r_w[0]; Z = t * r_w[2]
        D = float(np.hypot(X, Z))
        if not np.isfinite(D) or D <= 0: continue
        dr = float(depth_rel[v, u])
        if not (np.isfinite(dr) and dr > 0): continue
        kD = anchor_target_mult * D
        Rg = math.sqrt(h*h + kD*kD)
        s_list.append(dr * Rg)

    if not s_list:
        return s_med, {"mode": "anchor", "note": "no_valid_anchor", "s_med": float(s_med)}

    s_anchor_raw = float(np.median(np.array(s_list, float)))
    s_final = float(anchor_gain * s_anchor_raw)
    dbg["anchors"] = len(s_list)
    dbg["s_anchor_raw"] = s_anchor_raw
    return s_final, dbg

# -------------------- 方案 B／自動微調 (k, gain) --------------------
def auto_tune_anchor(depth_rel, K, R, h, anchors, s_med,
                     k_grid=np.arange(0.95, 1.31, 0.01),
                     g_grid=np.arange(0.95, 1.21, 0.01)):
    """在三錨點上搜尋 (k, gain) 使錨點的 D_midas 與幾何 D 的中位誤差最小。"""
    Ki = np.linalg.inv(K)
    def ray_world(u, v):
        r_cam = Ki @ np.array([u, v, 1.0], float)
        r_cam /= (np.linalg.norm(r_cam) + 1e-12)
        return R.T @ r_cam

    # 預先蒐集 (dr_i, D_i)
    samples = []
    for (u, v) in anchors:
        r_w = ray_world(u, v)
        if r_w[1] >= -1e-9: continue
        t = -h / r_w[1]
        if t <= 0: continue
        X = t * r_w[0]; Z = t * r_w[2]
        D = float(np.hypot(X, Z))
        dr = float(depth_rel[v, u])
        if not (np.isfinite(D) and D > 0 and np.isfinite(dr) and dr > 0): continue
        samples.append((dr, D))

    if len(samples) < 1:
        return None, {"note": "no_valid_anchor_for_auto"}

    # 給定 (k, g) 的誤差與 s
    def loss_for(k, g):
        s_raws = [dr * math.sqrt(h*h + (k*D)**2) for (dr, D) in samples]
        s = g * np.median(s_raws)
        preds = []
        for (dr, D) in samples:
            Rg = s / dr
            val = Rg*Rg - h*h
            if val <= 0: continue
            Dm = math.sqrt(val)
            preds.append(abs(Dm - D))
        if not preds:
            return float("inf"), s
        return float(np.median(preds)), s

    best = (float("inf"), None, None, None)  # (loss, s, k, g)
    for k in k_grid:
        for g in g_grid:
            L, s_candidate = loss_for(k, g)
            if L < best[0]:
                best = (L, s_candidate, k, g)

    L, s_best, k_best, g_best = best
    return (s_best if s_best is not None else s_med), {
        "k": k_best, "gain": g_best, "mad_error": L, "anchors": len(samples)
    }

# -------------------- JSON 讀取（多種編碼 fallback） --------------------
def _load_json_any_encoding(path):
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                data = json.load(f)
            # print(f"[INFO] JSON 用編碼 {enc} 讀取成功：{path}")
            return data
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except json.JSONDecodeError as e:
            raise ValueError(f"檔案可以用 {enc} 解碼，但不是合法 JSON：{e}")
    raise ValueError(f"無法用下列編碼解碼 JSON：{encodings}，最後錯誤：{last_err}")

def load_red_point_from_json_for_frame(json_path, frame_name):
    """
    專門處理：
    - accident_directions.json: list[ { "frame": "...jpg", "center": [u,v], ... }, ... ]
    - 或 frame_15.0s.json 這種：
        {
          "frame": "...jpg",
          "detections": [...],
          "tracks": { "1": [frame_idx, [u, v]], ... }
        }

    回傳: (u, v, frame_used)
    """
    data = _load_json_any_encoding(json_path)

    # case 1: list 型 (accident_directions.json)
    if isinstance(data, list):
        # 先嘗試 frame 完全匹配
        for item in data:
            if not isinstance(item, dict):
                continue
            if "frame" in item and item["frame"] == frame_name and "center" in item:
                c = item["center"]
                return float(c[0]), float(c[1]), item["frame"]

        # 找不到對應 frame，就拿第一筆有 center 的當 fallback
        for item in data:
            if isinstance(item, dict) and "center" in item:
                c = item["center"]
                frame_used = item.get("frame", frame_name)
                return float(c[0]), float(c[1]), frame_used

        raise ValueError(f"JSON(list) 裡找不到 frame={frame_name} 的 center 欄位。")

    # case 2: dict 型 (frame_15.0s.json / 單一檔)
    if isinstance(data, dict):
        frame_used = data.get("frame", frame_name)

        # 優先：如果有 "center" 欄位就直接用
        if "center" in data:
            c = data["center"]
            return float(c[0]), float(c[1]), frame_used

        # 例如：
        # {
        #   "frame": "frame_15.0s.jpg",
        #   "detections": [...],
        #   "tracks": { "1": [ 1, [373, 85] ] }
        # }
        if "tracks" in data and isinstance(data["tracks"], dict) and data["tracks"]:
            first_key = sorted(data["tracks"].keys(), key=lambda x: str(x))[0]
            val = data["tracks"][first_key]
            if isinstance(val, (list, tuple)) and len(val) == 2 and isinstance(val[1], (list, tuple)) and len(val[1]) == 2:
                u, v = float(val[1][0]), float(val[1][1])
                return u, v, frame_used

        raise ValueError(f"dict JSON 缺少 center 或 tracks 座標資訊：keys={list(data.keys())}")

    raise ValueError(f"無法解析 JSON（型態={type(data)}）來取得紅點座標。")

def append_final_record(video_name, distance_m, road_direction, camera_heading,
                        out_path="final_results.json"):
    import json, os, datetime

    record = {
        "video": video_name,
        "distance_m": float(distance_m),
        "road_direction": road_direction,
        "camera_heading": camera_heading,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []

    data.append(record)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    #print(f"[JSON] final record appended → {out_path}")


# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None, help="folder with frames")
    ap.add_argument("--glob", type=str, default=None, help='e.g. "C:\\path\\*.jpg"')
    ap.add_argument("--summary", type=str, default=None,
                    help="path to height_est_summary.json (from your car-track script)")
    ap.add_argument("--h", type=float, default=None, help="camera height (m), if no summary")
    ap.add_argument("--vp-relax", dest="vp_relax", action="store_true", help="relaxed VP thresholds")
    ap.add_argument("--scale-mode", type=str, default="anchor", choices=["median","anchor","blend"],
                    help="how to set s: median/anchor/blend (default: anchor)")
    ap.add_argument("--anchor-kind", type=str, default="three", choices=["center","three"],
                    help="anchor pixel set: center or three (default: three)")
    ap.add_argument("--anchor-target-mult", type=float, default=1.15,
                    help="multiply geometric D at anchors before anchoring (default: 1.15)")
    ap.add_argument("--anchor-gain", type=float, default=1.10,
                    help="final multiplicative gain after anchoring (default: 1.10)")
    ap.add_argument("--anchor-auto", action="store_true",
                    help="auto-tune k(anchor-target-mult) and gain via small grid search")
    ap.add_argument("--blend-alpha", type=float, default=0.7, help="alpha for blend mode")
    args = ap.parse_args()

    # 收集影像（MiDaS 用的 frame）
    paths = []
    if args.glob:
        paths = sorted(glob.glob(args.glob))
    elif args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg"))) + \
                sorted(glob.glob(os.path.join(args.dir, "*.png")))
    if not paths:
        raise SystemExit("No images found. Use --dir or --glob.")
    frames = [cv2.imread(p) for p in paths]
    frames = [im for im in frames if im is not None]
    if not frames:
        raise SystemExit("Failed to read any image.")

    first_path = paths[0]
    first = frames[0]
    H_img, W_img = first.shape[:2]
    cx, cy = W_img / 2.0, H_img / 2.0
    first_name = os.path.basename(first_path)
    # print(f"[INFO] frames loaded: {len(frames)}  size={W_img}x{H_img}  first={first_name}")

    # 這支程式所在資料夾 (例如 C:\Users\USER\Desktop\tst)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ========= 顯示用影像 disp：一律用 bbox/images 下「最後一張」 =========
    disp = first.copy()
    disp_name = first_name
    try:
        bbox_images_dir = os.path.join(base_dir, "bbox", "images")
        bbox_imgs = sorted(
            glob.glob(os.path.join(bbox_images_dir, "*.jpg")) +
            glob.glob(os.path.join(bbox_images_dir, "*.png"))
        )

        if bbox_imgs:
            # ★ 這裡改成：永遠取「排序後最後一張」作為事故定位圖
            chosen = bbox_imgs[-1]
            bbox_img = cv2.imread(chosen)
            if bbox_img is not None and bbox_img.shape[:2] == first.shape[:2]:
                disp = bbox_img.copy()
                disp_name = os.path.basename(chosen)
                # print(f"[INFO] 顯示用影像改成 bbox/images 最後一張：{chosen}")
            else:
                print(f"[WARN] bbox 圖尺寸 {None if bbox_img is None else bbox_img.shape[:2]} 與 MiDaS frame {first.shape[:2]} 不同，先用原始 frame。")
        else:
            print(f"[INFO] bbox/images 下沒有任何圖，顯示原始 frame。")
    except Exception as e:
        print(f"[WARN] 嘗試載入 bbox 圖時出錯：{e}，顯示原始 frame。")

    # ========= 相機內外參 =========
    if args.summary and os.path.isfile(args.summary):
        with open(args.summary, "r", encoding="utf-8") as f:
            S = json.load(f)
        h = float(S["h_estimate_m"])
        vpR = np.array([float(S["vp_road"][0]), float(S["vp_road"][1]), 1.0], float)
        vpV = np.array([float(S["vp_vert"][0]), float(S["vp_vert"][1]), 1.0], float)
        f0 = float(S["focal_px"])
        # print(f"[INFO] summary JSON loaded:")
        # print(f"       frames={S.get('frames')}, size={S.get('image_size')}, f≈{f0:.2f}, h≈{h:.3f}")
        # print(f"       vp_road=({vpR[0]:.3f},{vpR[1]:.3f})  vp_vert=({vpV[0]:.3f},{vpV[1]:.3f})")

        K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1.0]], float)
        R = solve_R_from_two_vps_with_known_K(vpR, vpV, K)
        R = orient_R_for_ground(K, R, W_img, H_img)

        Dbc, (u0, v0), frac, behind_qc = qc_bottom_center_range(K, R, h, W_img, H_img)
        if Dbc is None:
            print("[QC] bottom-center geom range: n/a (no forward ground intersection near bottom)")
        else:
            tail = " (behind camera!)" if behind_qc else ""
            # print(
                # f"[QC] bottom-center geom range ≈ {Dbc:.3f} m at v≈{(frac if frac is not None else float('nan')):.2f}H{tail} "
                # f"(JSON ~{S.get('ground_range_bottom_center_m', 'n/a')})"
            # )

        f_from_vps = focal_from_orthogonal_vps(vpR, vpV, cx, cy)
        # print(f"[INFO] sanity f_from_vps={f_from_vps:.2f}px (for info only)")
    else:
        if args.h is None:
            raise SystemExit("No --summary and no --h provided.")
        h = float(args.h)
        c1 = (60, 160)
        lines = detect_lines(first, canny1=c1[0], canny2=c1[1],
                             min_len=(60 if not args.vp_relax else 40),
                             gap=(6 if not args.vp_relax else 8))
        fam_road, fam_vert = split_two_families(lines, angle_tol_deg=(12 if not args.vp_relax else 18))
        if len(fam_road) < 2 or len(fam_vert) < 3:
            raise SystemExit("Not enough lines for vanishing points. Try --vp-relax or a clearer frame.")
        vpR = intersect_lines_ls(fam_road)
        vpV = intersect_lines_ls(fam_vert)
        K, R, f = solve_KR_from_two_vps(vpR, vpV, cx, cy)
        R = orient_R_for_ground(K, R, W_img, H_img)
        #print(f"[INFO] estimated f≈{f:.1f}px from VPs; using input h≈{h:.2f}m")

    # ========= MiDaS 深度 =========
    model, tfm, device = load_midas("MiDaS_small")
    depth_rel = fuse_depth_median(frames, model, tfm, device)
    #print("[INFO] median-depth fused.")

    # ========= 擬合 MiDaS 尺度 s =========
    s_med, sinfo = fit_midas_scale_stable(depth_rel, K, R, h,
                                          grid_step=6,
                                          x_range=(0.25, 0.75),
                                          y_range=(0.60, 0.95),
                                          trim=(10.0, 90.0),
                                          dmin=3.0, dmax=80.0)
    if s_med is None:
        # print(f"[WARN] could not fit MiDaS scale deterministically. info={sinfo}")
        s = None
    else:
        # print(f"[INFO] stable MiDaS scale (median) s_med≈{s_med:.3f}  (samples={sinfo['count']}, inliers={sinfo['inliers']})")

        if args.scale_mode == "anchor":
            anchors = _anchor_pixels(args.anchor_kind, W_img, H_img)
            if args.anchor_auto:
                s_auto, dbg_auto = auto_tune_anchor(depth_rel, K, R, h, anchors, s_med)
                if s_auto is not None:
                    s = s_auto
                    # print(f"[INFO] s (anchor-auto) → s≈{s:.3f} "
                          #f"[k*={dbg_auto['k']:.3f}, gain*={dbg_auto['gain']:.3f}, "
                          #f"MAD≈{dbg_auto['mad_error']:.3f}, anchors={dbg_auto['anchors']}]")"""
                else:
                    s, dbg = compute_s_with_anchor(depth_rel, K, R, h, s_med,
                                                   anchor_kind=args.anchor_kind,
                                                   anchor_target_mult=args.anchor_target_mult,
                                                   anchor_gain=args.anchor_gain,
                                                   W=W_img, H=H_img)
                    # print(f"[INFO] s (anchor fallback) → s≈{s:.3f} "
                          # f"[s_med={s_med:.3f}, k={args.anchor_target_mult:.3f}, gain={args.anchor_gain:.3f}]")
            else:
                s, dbg = compute_s_with_anchor(depth_rel, K, R, h, s_med,
                                               anchor_kind=args.anchor_kind,
                                               anchor_target_mult=args.anchor_target_mult,
                                               anchor_gain=args.anchor_gain,
                                               W=W_img, H=H_img)
        elif args.scale_mode == "blend":
            alpha = args.blend_alpha
            s_anchor, dbg = compute_s_with_anchor(depth_rel, K, R, h, s_med,
                                                  anchor_kind=args.anchor_kind,
                                                  anchor_target_mult=args.anchor_target_mult,
                                                  anchor_gain=args.anchor_gain,
                                                  W=W_img, H=H_img)
            if s_anchor is None:
                s = s_med
                #print(f"[INFO] s (blend) fallback to s_med≈{s_med:.3f}")
            else:
                s = float(alpha * s_med + (1.0 - alpha) * s_anchor)
                #print(f"[INFO] s (blend) → s≈{s:.3f}  [alpha={alpha:.2f}, s_med={s_med:.3f}, s_anchor={s_anchor:.3f}]")
        else:
            s = s_med
            #print(f"[INFO] s (median) → s≈{s:.3f}")

    # ========= 自動紅點量測：用「最後一張 bbox 圖」的檔名去找 json =========
    try:
        bbox_json_dir = os.path.join(base_dir, "bbox", "json")

        # 第一優先：與 disp_name (最後一張 bbox 圖) 同名的 json
        cand_paths = []
        frame_json_name = os.path.splitext(disp_name)[0] + ".json"
        cand_paths.append(os.path.join(bbox_json_dir, frame_json_name))
        # 第二優先：accident_directions.json
        cand_paths.append(os.path.join(bbox_json_dir, "accident_directions.json"))

        # 其他 json 當備用
        all_json = sorted(glob.glob(os.path.join(bbox_json_dir, "*.json")))
        for p in all_json:
            if p not in cand_paths:
                cand_paths.append(p)

        json_path = None
        for p in cand_paths:
            if os.path.isfile(p):
                json_path = p
                break

        if json_path is None:
            print(f"[INFO] bbox/json 裡找不到任何 JSON 檔，不做自動紅點量測。")
        else:
            # print(f"[INFO] 使用 JSON 紅點座標檔：{json_path}")
            try:
                u_red, v_red, frame_used = load_red_point_from_json_for_frame(json_path, disp_name)
                #print(f"[AUTO] 紅點像素位置 (u, v) ≈ ({u_red:.1f}, {v_red:.1f}) 對應 frame={frame_used}")

                # 如果 JSON 裡的 frame 跟 disp_name 不同，試著改用對應 bbox 圖
                if frame_used != disp_name:
                    alt_bbox_img_path = os.path.join(base_dir, "bbox", "images", frame_used)
                    if os.path.isfile(alt_bbox_img_path):
                        alt_img = cv2.imread(alt_bbox_img_path)
                        if alt_img is not None and alt_img.shape[:2] == first.shape[:2]:
                            disp = alt_img.copy()
                            disp_name = frame_used
                            #print(f"[INFO] 顯示改用 bbox/images/{frame_used}（與 JSON frame 對應）")
                        else:
                            print(f"[WARN] JSON frame 對應的 bbox 圖尺寸不符或無法讀取：{alt_bbox_img_path}")
                    else:
                        print(f"[WARN] 找不到 JSON frame 對應的 bbox 圖：{alt_bbox_img_path}")

                # ====== 計算幾何距離 + MiDaS 距離，選一個當最終距離 ======
                X, _, Z, D_geom, behind = ground_intersect(
                    int(round(u_red)), int(round(v_red)),
                    K, R, h, allow_backside=True
                )

                if X is None or D_geom is None:
                    print("[AUTO] 幾何光線與地面無交點，可能紅點在地平線以上。")
                else:
                    D_show = D_geom
                    method = "geom"

                    D_midas = None
                    if s is not None:
                        yy = int(round(v_red)); xx = int(round(u_red))
                        if 0 <= yy < depth_rel.shape[0] and 0 <= xx < depth_rel.shape[1]:
                            dr = float(depth_rel[yy, xx])
                            if dr > 0 and np.isfinite(dr):
                                Rg = s / dr
                                D_midas = math.sqrt(max(Rg*Rg - h*h, 0.0))
                                # 目前策略：若 MiDaS 有效，就用 MiDaS 當最終距離
                                D_show = D_midas
                                method = "midas"

                    msg = (f"[AUTO] px=({u_red:.1f},{v_red:.1f})  "
                           f"(X,Z)=({X:.3f},{Z:.3f}) m; D≈{D_show:.3f} m ({method})")
                    if behind:
                        msg += "  [behind camera]"
                    print(msg, flush=True)

                    try:
                        # 影片名稱：全專案都統一存 current_video.txt
                        video_name = open(os.path.join(base_dir, "current_video.txt")).read().strip()

                        # 車道方向：如果 accident_direction.json 有你的方向就讀
                        # 若沒有也不會出錯，可以先填 None
                        road_direction = None
                        lane_json = os.path.join(base_dir, "accident_direction.json")
                        if os.path.isfile(lane_json):
                            with open(lane_json, "r", encoding="utf-8") as f:
                                lane_info = json.load(f)
                                # 你可以改成 lane_info["accident_lane_direction"]
                                road_direction = lane_info.get("accident_lane_direction")

                        # camera heading：從 data.json 來（你原本 video_vlm 已經讀過）
                        camera_heading = None
                        data_json = os.path.join(base_dir, "data.json")
                        if os.path.isfile(data_json):
                            with open(data_json, "r", encoding="utf-8") as f:
                                arr = json.load(f)
                                for item in arr:
                                    if item.get("檔案名稱") == video_name:
                                        camera_heading = item.get("鏡頭拍攝方向")
                                        break

                        # 距離：用 D_show
                        append_final_record(
                            video_name,
                            D_show,
                            road_direction,
                            camera_heading,
                            out_path=os.path.join(base_dir, "final_results.json")
                        )

                    except Exception as e:
                        print(f"[WARN] JSON append failed: {e}")

                    # ====== 在圖上畫紅點 + 距離文字（在紅點左邊） ======
                    color = (0, 0, 255) if not behind else (0, 165, 255)
                    center_pt = (int(round(u_red)), int(round(v_red)))
                    cv2.circle(disp, center_pt, 7, color, -1)

                    label = f"D={D_show:.3f}m"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                    text_x = int(center_pt[0] - 10 - text_w)
                    text_y = int(center_pt[1] - 10)

                    text_x = max(text_x, 0)
                    text_y = max(text_y, text_h)

                    cv2.putText(disp, label, (text_x, text_y),
                                font, font_scale, color, thickness, cv2.LINE_AA)

                    cv2.namedWindow("red_point_distance", cv2.WINDOW_NORMAL)
                    cv2.imshow("red_point_distance", disp)
                    cv2.destroyAllWindows()
            except Exception as e_json:
                print(f"[WARN] 自動紅點量測失敗：{e_json}")

    except Exception as e:
        print(f"[WARN] 自動紅點量測時發生錯誤：{e}")


    print("[DONE] 距離計算完成")


if __name__ == "__main__":
    main()
