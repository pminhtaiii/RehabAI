import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import os
import argparse

def get_csv_path(env, base_dir=None):
    if base_dir:
        base = base_dir
    elif env == 'colab':
        base = '/content/drive/MyDrive/RehabAI'
    else:
        base = 'C:/RehabAI'
    return f'{base}/05_final_datasets/KiMoRe_data_movenet_features.csv'

def check_features(args):
    exercise = args.exercise
    csv_path = get_csv_path(args.env, args.base_dir)
    
    print(f"Loading data for {exercise} from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"LỖI: Không tìm thấy file CSV tại {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if exercise != 'All':
        df = df[df['exercise'] == exercise].reset_index(drop=True)
    
    df = df.dropna(subset=['clinical_score', 'joint_features']).reset_index(drop=True)
    
    X = []
    y = []
    skipped_paths = 0
    
    for _, row in df.iterrows():
        fpath = str(row['joint_features'])
        
        # Nếu path lưu trong CSV là đường dẫn C:/... nhưng đang chạy trên Colab,
        # cần phải replace lại cho đúng môi trường Colab.
        if args.env == 'colab' and 'C:/' in fpath:
            fpath = fpath.replace('C:/RehabAI', '/content/drive/MyDrive/RehabAI')
            fpath = fpath.replace('\\', '/')

        if not os.path.exists(fpath):
            if skipped_paths < 3:
                print(f"  [Cảnh báo] Không tìm thấy file features: {fpath}")
            skipped_paths += 1
            continue
            
        try:
            feat_df = pd.read_csv(fpath)
            feat_arr = feat_df.to_numpy(dtype=np.float64)
            if feat_arr.shape[0] < 5: 
                continue
            
            # Tính các đặc trưng tĩnh để feed cho Random Forest
            mean_vals = np.mean(feat_arr, axis=0)
            std_vals = np.std(feat_arr, axis=0)
            min_vals = np.min(feat_arr, axis=0)
            max_vals = np.max(feat_arr, axis=0)
            
            # Nối lại thành 1 vector
            static_vector = np.concatenate([mean_vals, std_vals, min_vals, max_vals])
            X.append(static_vector)
            y.append(float(row['clinical_score']))
            
        except Exception as e:
            print(f"  [Lỗi] khi đọc {fpath}: {e}")
            continue
            
    if skipped_paths > 0:
        print(f"Đã bỏ qua {skipped_paths} files vì không tìm thấy đường dẫn (File Not Found).")

    if len(X) == 0:
        print("\n❌ LỖI CRITICAL: Không load được bất kỳ video nào!")
        print("Biến X rỗng, gây ra lỗi IndexError: tuple index out of range.")
        print("Vui lòng kiểm tra lại xem đường dẫn trong cột `joint_features` có khớp với Colab không.")
        return
        
    X = np.nan_to_num(np.array(X))
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples successfully. Feature vector size: {X.shape[1]}")
    
    # Train Random Forest với K-Fold
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Dự đoán OOF (Out-of-fold)
    print("Training Random Forest to verify feature quality...")
    oof_preds = cross_val_predict(rf, X, y, cv=kf)
    
    mae = mean_absolute_error(y, oof_preds)
    spearman_rho, p = spearmanr(y, oof_preds)
    
    print("\n" + "="*50)
    print(f"Random Forest Diagnostic Results ({exercise}):")
    print(f"  MAE: {mae:.2f}")
    print(f"  Spearman ρ: {spearman_rho:.3f} (p={p:.4f})")
    print("="*50)
    print("\nKẾT LUẬN:")
    if spearman_rho > 0.3:
        print("✅ Features CÓ CHỨA thông tin hữu ích (Spearman > 0.3).")
        print("Mức độ này cho thấy file 03 trích xuất tương đối ổn định.")
    else:
        print("❌ Features BỊ NHIỄU QUÁ NHIỀU (Spearman rất thấp hoặc âm).")
        print("Nguyên nhân 99% do tính góc 3D bị sai tỷ lệ khung hình (Aspect Ratio)")
        print("hoặc trục Z của MediaPipe bị nhiễu loạn (không đáng tin cậy bằng Kinect).")
        print("👉 Chúng ta bắt buộc phải sửa đổi lại logic toán học ở file 03_extract_joint_features.py.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['colab', 'local'], default='colab')
    parser.add_argument('--exercise', default='Es1')
    parser.add_argument('--base-dir', default=None)
    args = parser.parse_args()
    
    check_features(args)
