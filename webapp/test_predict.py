import os
import requests

BASE = "http://127.0.0.1:5000"

print("--- Testing GET / ---")
r_index = requests.get(BASE + "/")
print(f"Status: {r_index.status_code}, HTML length: {len(r_index.text)}")

print("\n--- Testing GET /samples ---")
r_samples = requests.get(BASE + "/samples")
samples_data = r_samples.json()
print(f"Status: {r_samples.status_code}")
print("Samples count:", len(samples_data.get("samples", [])))
print("Samples:", samples_data.get("samples", []))

print("\n--- Testing GET /data/test_confusion_matrices.png ---")
r_img = requests.get(BASE + "/data/test_confusion_matrices.png")
print(f"Status: {r_img.status_code}, Image bytes: {len(r_img.content)}")

print("\n--- Testing GET /api/model-comparison ---")
r_comp = requests.get(BASE + "/api/model-comparison")
print(f"Status: {r_comp.status_code}")
for m in r_comp.json().get("metrics", []):
    print(f"  - {m['model']}: Accuracy={m['accuracy']}%, F1={m['f1_score']}")

print("\n--- Testing POST /predict on all 8 demo clips ---")
demo_files = [
    ("demo_samples/calm_1.wav", "Calm"),
    ("demo_samples/calm_2.wav", "Calm"),
    ("demo_samples/calm_3.wav", "Calm"),
    ("demo_samples/calm_4.wav", "Calm"),
    ("demo_samples/stressed_1.wav", "Stressed"),
    ("demo_samples/stressed_2.wav", "Stressed"),
    ("demo_samples/stressed_3.wav", "Stressed"),
    ("demo_samples/stressed_4.wav", "Stressed"),
]

for fpath, true_label in demo_files:
    fname = os.path.basename(fpath)
    with open(fpath, "rb") as f:
        r = requests.post(BASE + "/predict", files={"audio": f}, data={"true_label": true_label})
    if r.status_code == 200:
        d = r.json()
        match_str = "MATCH [OK]" if d.get("is_correct") else "MISMATCH"
        print(f"[{fname}] True={d.get('true_label')} | Pred={d.get('predicted_class')} ({d.get('stress_score')}%, {d.get('level')} Stress) | Baseline={d['per_model']['baseline']}% LSTM={d['per_model']['lstm']}% CNN={d['per_model']['cnn']}% | {match_str}")
    else:
        print(f"[{fname}] FAILED with code {r.status_code}: {r.text}")

print("\nAll API tests completed successfully!")
