"""
RehabAI MoveNet Model Test Suite — Self-contained for Google Colab
==================================================================
Chạy trên Colab: paste vào 1 cell và run.
Yêu cầu: mount Google Drive trước.

Script này dùng bộ feature extraction 2D (x,y) chuẩn MoveNet, KHÔNG dùng temporal stats.
Phù hợp 100% với model gốc (Es1=9, Es2=9, Es3=13, Es4=6, Es5=9 features).
"""
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = "/content/drive/MyDrive/RehabAI"
MODELS_DIR = f"{BASE_DIR}/models/best_models"
SCALERS_DIR = MODELS_DIR
DOWNSAMPLE = 5
MAX_LEN = {"Es1": 301, "Es2": 326, "Es3": 297, "Es4": 398, "Es5": 204}

COLS = []
for jt in ["left_shoulder","right_shoulder","left_elbow","right_elbow",
           "left_wrist","right_wrist","left_hip","right_hip",
           "left_knee","right_knee","left_ankle","right_ankle"]:
    COLS += [f"{jt}_x", f"{jt}_y", f"{jt}_z", f"{jt}_v"]

# ── 2D GEOMETRY ───────────────────────────────────────────────────────────────
def jp(df, name):
    return [df[f"{name}_x"], df[f"{name}_y"]]

def angle2d(first, middle, end):
    first, middle, end = np.array(first), np.array(middle), np.array(end)
    radians = np.arctan2(end[1]-middle[1], end[0]-middle[0]) - np.arctan2(first[1]-middle[1], first[0]-middle[0])
    angle = np.abs(radians*180.0/np.pi)
    return [360-a if a > 180 else a for a in angle]

def dist2d(p1, p2):
    return pd.Series([math.dist([x1,y1],[x2,y2]) for x1,y1,x2,y2 in zip(p1[0],p1[1],p2[0],p2[1])])

# ── FEATURE EXTRACTORS ────────────────────────────────────────────────────────
def get_es1(df):
    f = pd.DataFrame()
    f["la_torso"] = angle2d(jp(df,"left_elbow"), jp(df,"left_shoulder"), jp(df,"left_hip"))
    f["ra_torso"] = angle2d(jp(df,"right_elbow"), jp(df,"right_shoulder"), jp(df,"right_hip"))
    f["le_ext"] = angle2d(jp(df,"left_shoulder"), jp(df,"left_elbow"), jp(df,"left_wrist"))
    f["re_ext"] = angle2d(jp(df,"right_shoulder"), jp(df,"right_elbow"), jp(df,"right_wrist"))
    f["lk_ext"] = angle2d(jp(df,"left_hip"), jp(df,"left_knee"), jp(df,"left_ankle"))
    f["rk_ext"] = angle2d(jp(df,"right_hip"), jp(df,"right_knee"), jp(df,"right_ankle"))
    mhp = [(jp(df,"left_hip")[0]+jp(df,"right_hip")[0])/2, (jp(df,"left_hip")[1]+jp(df,"right_hip")[1])/2]
    f["hip_angle"] = angle2d(jp(df,"left_hip"), mhp, jp(df,"right_hip"))
    f["hands_dist"] = dist2d(jp(df,"left_wrist"), jp(df,"right_wrist"))
    f["ankle_dist"] = dist2d(jp(df,"left_ankle"), jp(df,"right_ankle"))
    return f

def get_es2(df):
    f = pd.DataFrame()
    f["le_ext"] = angle2d(jp(df,"left_shoulder"), jp(df,"left_elbow"), jp(df,"left_wrist"))
    f["re_ext"] = angle2d(jp(df,"right_shoulder"), jp(df,"right_elbow"), jp(df,"right_wrist"))
    f["lk_ext"] = angle2d(jp(df,"left_hip"), jp(df,"left_knee"), jp(df,"left_ankle"))
    f["rk_ext"] = angle2d(jp(df,"right_hip"), jp(df,"right_knee"), jp(df,"right_ankle"))
    mhp = [(jp(df,"left_hip")[0]+jp(df,"right_hip")[0])/2, (jp(df,"left_hip")[1]+jp(df,"right_hip")[1])/2]
    f["hip_angle"] = angle2d(jp(df,"left_hip"), mhp, jp(df,"right_hip"))
    f["hands_dist"] = dist2d(jp(df,"left_wrist"), jp(df,"right_wrist"))
    f["shoulder_dist"] = dist2d(jp(df,"left_shoulder"), jp(df,"right_shoulder"))
    f["lsw_vd"] = np.abs(df["left_shoulder_y"]-df["left_wrist_y"])
    f["rsw_vd"] = np.abs(df["right_shoulder_y"]-df["right_wrist_y"])
    return f

def get_es3(df):
    f = pd.DataFrame()
    f["elbow_hd"] = np.abs(df["left_elbow_x"]-df["right_elbow_x"])
    f["le_ext"] = angle2d(jp(df,"left_shoulder"), jp(df,"left_elbow"), jp(df,"left_wrist"))
    f["re_ext"] = angle2d(jp(df,"right_shoulder"), jp(df,"right_elbow"), jp(df,"right_wrist"))
    f["lk_ext"] = angle2d(jp(df,"left_hip"), jp(df,"left_knee"), jp(df,"left_ankle"))
    f["rk_ext"] = angle2d(jp(df,"right_hip"), jp(df,"right_knee"), jp(df,"right_ankle"))
    f["ls_ext"] = angle2d(jp(df,"left_elbow"), jp(df,"left_shoulder"), jp(df,"right_shoulder"))
    f["rs_ext"] = angle2d(jp(df,"right_elbow"), jp(df,"right_shoulder"), jp(df,"left_shoulder"))
    mhp = [(jp(df,"left_hip")[0]+jp(df,"right_hip")[0])/2, (jp(df,"left_hip")[1]+jp(df,"right_hip")[1])/2]
    f["hip_angle"] = angle2d(jp(df,"left_hip"), mhp, jp(df,"right_hip"))
    f["hands_dist"] = dist2d(jp(df,"left_wrist"), jp(df,"right_wrist"))
    f["shoulder_dist"] = dist2d(jp(df,"left_shoulder"), jp(df,"right_shoulder"))
    f["hip_dist"] = dist2d(jp(df,"left_hip"), jp(df,"right_hip"))
    f["lsw_vd"] = np.abs(df["left_shoulder_y"]-df["left_wrist_y"])
    f["rsw_vd"] = np.abs(df["right_shoulder_y"]-df["right_wrist_y"])
    return f

def get_es4(df):
    f = pd.DataFrame()
    f["le_ext"] = angle2d(jp(df,"left_shoulder"), jp(df,"left_elbow"), jp(df,"left_wrist"))
    f["re_ext"] = angle2d(jp(df,"right_shoulder"), jp(df,"right_elbow"), jp(df,"right_wrist"))
    f["lk_ext"] = angle2d(jp(df,"left_hip"), jp(df,"left_knee"), jp(df,"left_ankle"))
    f["rk_ext"] = angle2d(jp(df,"right_hip"), jp(df,"right_knee"), jp(df,"right_ankle"))
    f["shoulder_dist"] = dist2d(jp(df,"left_shoulder"), jp(df,"right_shoulder"))
    f["hip_dist"] = dist2d(jp(df,"left_hip"), jp(df,"right_hip"))
    return f

def get_es5(df):
    f = pd.DataFrame()
    f["lk_ext"] = angle2d(jp(df,"left_hip"), jp(df,"left_knee"), jp(df,"left_ankle"))
    f["rk_ext"] = angle2d(jp(df,"right_hip"), jp(df,"right_knee"), jp(df,"right_ankle"))
    f["hands_dist"] = dist2d(jp(df,"left_wrist"), jp(df,"right_wrist"))
    f["shoulder_dist"] = dist2d(jp(df,"left_shoulder"), jp(df,"right_shoulder"))
    f["hip_dist"] = dist2d(jp(df,"left_hip"), jp(df,"right_hip"))
    f["knee_dist"] = dist2d(jp(df,"left_knee"), jp(df,"right_knee"))
    f["ankle_dist"] = dist2d(jp(df,"left_ankle"), jp(df,"right_ankle"))
    f["lsw_dist"] = dist2d(jp(df,"left_shoulder"), jp(df,"left_wrist"))
    f["rsw_dist"] = dist2d(jp(df,"right_shoulder"), jp(df,"right_wrist"))
    return f

EXTRACTORS = {"Es1":get_es1,"Es2":get_es2,"Es3":get_es3,"Es4":get_es4,"Es5":get_es5}

# ── PIPELINE ──────────────────────────────────────────────────────────────────
def prepare(df, ex_id, max_len):
    """Keypoints → 2D features → scale → downsample → pad"""
    df = df.head(600).fillna(0.0)
    feat = EXTRACTORS[ex_id](df).to_numpy(dtype=np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    sc_path = os.path.join(SCALERS_DIR, f"scaler_{ex_id}.joblib")
    if os.path.exists(sc_path):
        scaler = joblib.load(sc_path)
        if feat.shape[1] == scaler.n_features_in_:
            feat = scaler.transform(feat).astype(np.float32)
        else:
            print(f"  [WARN] {ex_id}: scaler expects {scaler.n_features_in_}, got {feat.shape[1]}")
    
    # Downsample frames
    feat = feat[::DOWNSAMPLE]
    
    # Pad/truncate sequences
    if feat.shape[0] > max_len: 
        feat = feat[:max_len]
    elif feat.shape[0] < max_len:
        feat = np.pad(feat, ((0,max_len-feat.shape[0]),(0,0)), constant_values=0.0)
        
    return np.expand_dims(np.nan_to_num(feat.astype(np.float32)), axis=0)

# ── TEST CASES ────────────────────────────────────────────────────────────────
def make_pose(n=300):
    template = {
        "left_shoulder":(0.40,0.25,-0.05),"right_shoulder":(0.60,0.25,-0.05),
        "left_elbow":(0.35,0.35,-0.03),"right_elbow":(0.65,0.35,-0.03),
        "left_wrist":(0.30,0.45,-0.02),"right_wrist":(0.70,0.45,-0.02),
        "left_hip":(0.43,0.50,-0.01),"right_hip":(0.57,0.50,-0.01),
        "left_knee":(0.43,0.70,0.00),"right_knee":(0.57,0.70,0.00),
        "left_ankle":(0.43,0.90,0.01),"right_ankle":(0.57,0.90,0.01),
    }
    arr = np.zeros((n, len(COLS)), dtype=np.float32)
    for jt,(x,y,z) in template.items():
        arr[:,COLS.index(f"{jt}_x")] = x
        arr[:,COLS.index(f"{jt}_y")] = y
        arr[:,COLS.index(f"{jt}_z")] = z
        arr[:,COLS.index(f"{jt}_v")] = 0.9
    return arr

def ci(name): return COLS.index(name)

def gen_tests():
    n = 300
    cases = {}
    
    # Static & Error Cases
    cases["1_All_Zeros"] = np.zeros((n, len(COLS)), np.float32)
    cases["2_Standing_Still"] = make_pose(n) + np.random.normal(0,0.001,(n,len(COLS))).astype(np.float32)
    cases["3_NaN_Input"] = np.full((n, len(COLS)), np.nan, np.float32)
    cases["4_Huge_Values"] = make_pose(n) * 1000
    cases["5_Negative_Coords"] = make_pose(n) * -1
    cases["6_Single_Frame"] = make_pose(1)
    cases["7_Very_Short_5f"] = make_pose(5)
    
    # Es1: Arm raises
    p = make_pose(n)
    for i in range(n):
        a = 0.25*np.sin(2*np.pi*i/n)
        p[i,ci("left_wrist_y")] -= a; p[i,ci("right_wrist_y")] -= a
        p[i,ci("left_elbow_y")] -= a*0.5; p[i,ci("right_elbow_y")] -= a*0.5
    cases["8_Es1_ArmRaise"] = p
    
    # Es2: Elbow flexion
    p = make_pose(n)
    for i in range(n):
        a = 0.15*np.sin(2*np.pi*i/n)
        p[i,ci("left_wrist_y")] -= a; p[i,ci("left_wrist_x")] -= a*0.3
        p[i,ci("right_wrist_y")] -= a; p[i,ci("right_wrist_x")] += a*0.3
    cases["9_Es2_ElbowFlex"] = p
    
    # Es3: Shoulder rotation
    p = make_pose(n)
    for i in range(n):
        a = 0.12*np.sin(2*np.pi*i/n)
        p[i,ci("left_elbow_x")] -= a; p[i,ci("right_elbow_x")] += a
    cases["10_Es3_ShoulderRot"] = p
    
    # Es4: Squat
    p = make_pose(n)
    for i in range(n):
        d = 0.15*max(0,np.sin(2*np.pi*i/n))
        p[i,ci("left_hip_y")] += d; p[i,ci("right_hip_y")] += d
        p[i,ci("left_knee_y")] += d*0.5; p[i,ci("right_knee_y")] += d*0.5
    cases["11_Es4_Squat"] = p
    
    # Es5: Sit-to-stand
    p = make_pose(n)
    for i in range(n):
        d = 0.2*max(0,np.sin(2*np.pi*i/n))
        p[i,ci("left_hip_y")] += d; p[i,ci("right_hip_y")] += d
        p[i,ci("left_knee_y")] += d*0.8; p[i,ci("right_knee_y")] += d*0.8
        p[i,ci("left_shoulder_y")] += d*0.3; p[i,ci("right_shoulder_y")] += d*0.3
    cases["12_Es5_SitToStand"] = p
    
    return cases

# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
def run():
    tests = gen_tests()
    all_results = {}
    
    print(f"Bắt đầu chạy kiểm thử MoveNet models trong {MODELS_DIR}...")
    
    for ex_id in ["Es1","Es2","Es3","Es4","Es5"]:
        ml = MAX_LEN[ex_id]
        mpath = os.path.join(MODELS_DIR, f"ml_model_{ex_id}_best.keras")
        if not os.path.exists(mpath):
            print(f"❌ {ex_id}: not found at {mpath}")
            continue
            
        model = tf.keras.models.load_model(mpath, compile=False)
        print(f"\n{'='*50}\n  Model: {ex_id} | input={model.input_shape}\n{'='*50}")
        
        ex_res = {}
        for name, arr in tests.items():
            try:
                df = pd.DataFrame(arr, columns=COLS)
                data = prepare(df, ex_id, ml)
                pred = model.predict(data, verbose=0).flatten()
                score = round(float(pred[0]*100), 2)
                ex_res[name] = {"score": score, "status": "OK"}
                print(f"  {name.ljust(25)} → {score:7.2f}")
            except Exception as e:
                ex_res[name] = {"score": "ERROR", "status": str(e)[:80]}
                print(f"  {name.ljust(25)} → ERROR: {str(e)[:80]}")
        all_results[ex_id] = ex_res

    # ── Tạo Báo Cáo Markdown ──
    desc = {
        "1_All_Zeros":"Toàn bộ input = 0 (không có người)",
        "2_Standing_Still":"Đứng yên, noise rất nhỏ",
        "3_NaN_Input":"Input toàn NaN (lỗi camera)",
        "4_Huge_Values":"Tọa độ cực lớn x1000 (lỗi dữ liệu)",
        "5_Negative_Coords":"Tọa độ âm (lỗi tracking)",
        "6_Single_Frame":"Chỉ có 1 frame duy nhất",
        "7_Very_Short_5f":"Chỉ có 5 frames",
        "8_Es1_ArmRaise":"Giơ tay lên xuống (bài tập Es1)",
        "9_Es2_ElbowFlex":"Co duỗi khuỷu tay (bài tập Es2)",
        "10_Es3_ShoulderRot":"Xoay vai (bài tập Es3)",
        "11_Es4_Squat":"Ngồi xổm (bài tập Es4)",
        "12_Es5_SitToStand":"Đứng lên ngồi xuống (bài tập Es5)",
    }
    
    md = "# 📊 RehabAI — Kết quả kiểm thử mô hình MoveNet\n\n"
    md += "## Mô tả Test Cases\n\n| # | Test Case | Mô tả |\n|---|---|---|\n"
    for k,v in desc.items(): 
        md += f"| {k.split('_')[0]} | {k} | {v} |\n"
        
    md += "\n## Kết quả theo từng mô hình\n\n"
    for ex_id, ex_res in all_results.items():
        md += f"### Mô hình {ex_id}\n\n| Test Case | Điểm (0-100) | Trạng thái |\n|---|---|---|\n"
        for name, r in ex_res.items():
            s = r["score"] if r["status"]=="OK" else f'ERROR: {r["status"]}'
            icon = "✅" if r["status"]=="OK" else "❌"
            md += f"| {name} | {s} | {icon} |\n"
        md += "\n"
        
    # Phân tích đánh giá
    md += "## 🔍 Phân tích tự động\n\n"
    for ex_id, ex_res in all_results.items():
        md += f"### {ex_id}\n"
        scores = {k:v["score"] for k,v in ex_res.items() if v["status"]=="OK"}
        if not scores: 
            md += "- ❌ Không có kết quả hợp lệ\n\n"
            continue
            
        static = [scores[k] for k in ["1_All_Zeros","2_Standing_Still"] if k in scores]
        ek = {"Es1":"8_Es1_ArmRaise","Es2":"9_Es2_ElbowFlex","Es3":"10_Es3_ShoulderRot","Es4":"11_Es4_Squat","Es5":"12_Es5_SitToStand"}
        ex_score = scores.get(ek[ex_id])
        
        if static and ex_score is not None:
            avg_s = np.mean(static)
            diff = ex_score - avg_s
            if abs(diff) < 2: 
                md += f"- ⚠️ **Không phân biệt được** chuyển động vs đứng yên (chênh lệch {diff:.1f})\n"
            elif diff > 0: 
                md += f"- ✅ Chuyển động ({ex_score:.1f}) > đứng yên ({avg_s:.1f}), chênh lệch {diff:.1f}\n"
            else: 
                md += f"- 🔄 Chuyển động ({ex_score:.1f}) < đứng yên ({avg_s:.1f}), chênh lệch {diff:.1f}\n"
                
        vals = [v for v in scores.values() if isinstance(v,(int,float))]
        if vals:
            md += f"- Phạm vi điểm số: [{min(vals):.1f} — {max(vals):.1f}], Biên độ (Spread): {max(vals)-min(vals):.1f}\n"
            if max(vals) - min(vals) < 5: 
                md += f"- ⚠️ **Spread quá nhỏ** — mô hình có thể đang collapse về trung bình\n"
                
        errors = [k for k,v in ex_res.items() if v["status"]!="OK"]
        if errors: 
            md += f"- ❌ Lỗi ở các test: {', '.join(errors)}\n"
        md += "\n"

    out_path = os.path.join(MODELS_DIR, "test_results_movenet.md")
    with open(out_path, "w", encoding="utf-8") as f: 
        f.write(md)
    print(f"\n✅ Báo cáo đã lưu tại: {out_path}")

if __name__ == "__main__":
    run()
