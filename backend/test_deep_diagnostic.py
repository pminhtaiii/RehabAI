"""
Deep diagnostic: check model ↔ scaler ↔ backend compatibility.
Sends requests to running backend and analyzes results.
No tensorflow required.
"""
import urllib.request
import json
import sys

BASE_URL = "http://localhost:8000"

def fetch(endpoint, timeout=120):
    try:
        with urllib.request.urlopen(f"{BASE_URL}{endpoint}", timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("  RehabAI Deep Diagnostic — Model ↔ Scaler ↔ Backend Compatibility")
    print("=" * 70)

    # 1. Health check
    health = fetch("/api/health", timeout=10)
    if "error" in health:
        print(f"\n  ✗ Backend not reachable: {health['error']}")
        sys.exit(1)
    print("\n  ✓ Backend is running")
    
    # Show which models loaded
    for ex, info in health.get("models", {}).items():
        loaded = info.get("model_loaded", False)
        err = info.get("error")
        if loaded:
            print(f"    ✓ {ex}: loaded")
        else:
            print(f"    ✗ {ex}: FAILED — {err}")

    exercises = ["Es1", "Es2", "Es3", "Es4", "Es5"]
    
    for ex in exercises:
        print(f"\n{'─' * 70}")
        print(f"  DIAGNOSING: {ex}")
        print(f"{'─' * 70}")
        
        # 2. Run diagnostic
        print(f"\n  [1] Diagnostic endpoint (3 test cases)...")
        diag = fetch(f"/api/diagnostic/{ex}")
        if "error" in diag:
            print(f"    ✗ Error: {diag['error']}")
            continue
        
        shape = diag.get("model_input_shape", "?")
        print(f"    Model input shape: {shape}")
        
        raw = diag.get("raw_scores", {})
        cal = diag.get("calibrated_scores", {})
        raw_range = diag.get("raw_score_range", 0)
        responsive = diag.get("model_is_responsive", False)
        
        print(f"    Raw scores:  no_move={raw.get('no_movement','?')}, "
              f"standing={raw.get('standing_still','?')}, "
              f"arm_raise={raw.get('arm_raise','?')}")
        print(f"    Cal scores:  no_move={cal.get('no_movement','?')}, "
              f"standing={cal.get('standing_still','?')}, "
              f"arm_raise={cal.get('arm_raise','?')}")
        print(f"    Raw range:   {raw_range:.1f}")
        
        # 3. Analyze raw model outputs
        # reverse-engineer: score = clip(raw_out * 50 or raw_out, 0, 50) * 2
        print(f"\n  [2] Raw model output analysis:")
        for test_name, score in raw.items():
            if score >= 99.5:
                print(f"    {test_name}: score={score} → model output ≥ 1.0 (CLIPPED to max)")
                print(f"      → Possible: model NOT v6 (no sigmoid), or collapsed upward")
            elif score <= 0.5:
                print(f"    {test_name}: score={score} → model output ≈ 0 (CLIPPED to min)")
            else:
                # auto-detect: if raw_out <= 1.5, we used raw_out * 50
                # score = clip(raw_out * 50, 0, 50) * 2
                # raw_out = score / 2 / 50
                raw_out = score / 2.0 / 50.0
                if raw_out <= 1.0:
                    print(f"    {test_name}: score={score:.1f} → model output ≈ {raw_out:.4f} "
                          f"(sigmoid range ✓)")
                else:
                    print(f"    {test_name}: score={score:.1f} → model output ≈ {raw_out:.4f} "
                          f"(outside sigmoid range!)")
        
        # 4. Check for mean collapse
        mean_score = sum(raw.values()) / len(raw) if raw else 0
        all_similar = raw_range < 5.0
        print(f"\n  [3] Collapse analysis:")
        if all_similar:
            print(f"    ⚠️  ALL scores within {raw_range:.1f} points → MEAN COLLAPSE")
            print(f"    Mean score: {mean_score:.1f}")
            if 40 <= mean_score <= 60:
                print(f"    → Model predicting dataset mean (~{mean_score:.0f}/100)")
            elif mean_score >= 95:
                print(f"    → Model saturated at max (sigmoid saturation or no sigmoid)")
        else:
            print(f"    ✅ Model discriminates (range={raw_range:.1f})")
        
        # 5. Motion analysis
        motion = diag.get("motion_analysis", {})
        print(f"\n  [4] Motion detection:")
        for test_name, m in motion.items():
            if isinstance(m, dict):
                active = m.get("active", "?")
                energy = m.get("energy", 0)
                qf = m.get("quality_factor", 0)
                print(f"    {test_name}: active={active}, energy={energy:.3f}, qf={qf:.3f}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    
    for ex in exercises:
        diag = fetch(f"/api/diagnostic/{ex}")
        if "error" in diag:
            print(f"  ✗ {ex}: error")
            continue
        
        raw = diag.get("raw_scores", {})
        raw_range = diag.get("raw_score_range", 0)
        shape = diag.get("model_input_shape", "?")
        
        # Parse shape to get n_features
        # shape like "(None, 150, 6)"
        try:
            parts = shape.replace("(", "").replace(")", "").split(",")
            n_features = int(parts[-1].strip())
            max_len = int(parts[1].strip())
        except:
            n_features = "?"
            max_len = "?"
        
        mean_score = sum(raw.values()) / len(raw) if raw else 0
        
        status = "✅" if raw_range > 5 else ("⚠️ " if raw_range > 2 else "❌")
        collapse_note = ""
        if raw_range < 5:
            collapse_note = f" (mean={mean_score:.0f}, COLLAPSED)"
        
        print(f"  {status} {ex}: shape=({max_len},{n_features}) | "
              f"range={raw_range:.1f}{collapse_note}")
    
    print(f"\n  LEGEND:")
    print(f"  - If model shape has WRONG n_features → scaler mismatch")
    print(f"  - If all scores = 100 → model has no sigmoid (v5 style)")  
    print(f"  - If range < 5 → model collapsed to mean")
    print(f"  - If range > 5 → model discriminates inputs")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
