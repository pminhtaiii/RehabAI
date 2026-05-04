"""Check raw model output for Es3 to determine if it was trained with y/50 normalization."""

import urllib.request
import json

url = "http://localhost:8000/api/diagnostic/Es3"
print("Fetching Es3 diagnostic (may take ~30s)...")
try:
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read().decode())
except Exception as e:
    print(f"Error: {e}")
    exit(1)

print(f"\nModel input shape: {data.get('model_input_shape')}")
print(f"\nRaw scores (after *50 un-normalization + *2 scaling):")
print(f"  {data.get('raw_scores')}")

# The diagnostic endpoint also stores the direct model output in motion_analysis
# But more importantly, we can reverse-engineer the raw model output:
# raw_score = clip(model_output * 50, 0, 50) * 2
# So: model_output = raw_score / 2 / 50

for test_name, score in data.get("raw_scores", {}).items():
    if score == 100.0:
        print(f"\n  {test_name}: score=100 → model_output ≥ 1.0")
        print(f"    If v5 trained: model collapsed (outputs ≥1.0, clipped to max)")
        print(f"    If NOT v5: model outputs ~25-50 in [0,50] range, *50 pushes to 100")
    else:
        model_out = score / 2.0 / 50.0
        print(f"\n  {test_name}: score={score} → model_output ≈ {model_out:.4f}")

# Check Es1 for comparison
print("\n\nFetching Es1 diagnostic for comparison...")
try:
    with urllib.request.urlopen("http://localhost:8000/api/diagnostic/Es1", timeout=120) as resp:
        data1 = json.loads(resp.read().decode())
    print(f"Es1 raw scores: {data1.get('raw_scores')}")
    for test_name, score in data1.get("raw_scores", {}).items():
        model_out = score / 2.0 / 50.0
        print(f"  {test_name}: score={score:.1f} → model_output ≈ {model_out:.4f}")
except Exception as e:
    print(f"Error: {e}")

# Conclusion
print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)
print("If Es3 model_output ≥ 1.0 for ALL inputs:")
print("  → Either model collapsed to max, OR model was trained")
print("    WITHOUT y/50 normalization (outputs [0,50] directly)")
print("")
print("Compare with Es1 model_output (~0.2-0.75):")
print("  → Es1 clearly uses y/50 normalization (outputs in [0,1])")
print("")
print("If Es3 was trained without normalization:")
print("  → Fix: remove the *50 un-normalization for Es3 only")
print("  → Or: check model_config_Es3.json for 'version' field")
