"""Validate exported experiment records using only the Python standard library."""
from pathlib import Path
import csv
import hashlib
import json
import re

root = Path(__file__).resolve().parents[1]
summaries = json.loads((root / 'results/summary.json').read_text())
with (root / 'results/eval_history_all.csv').open(encoding='utf-8-sig') as f:
    records = list(csv.DictReader(f))
assert len(records) == 165
for s in summaries:
    n = s['dataset_size']
    rows = [r for r in records if int(r['dataset_size']) == n]
    assert [int(r['step']) for r in rows] == list(range(0, 257, 8))
    for r in rows:
        count = int(r['correct_count_reconstructed'])
        assert 0 <= count <= 1319
        assert abs(count / 1319 - float(r['accuracy_log_4dp'])) <= 0.000051
    best = max(float(r['accuracy_reconstructed']) for r in rows)
    assert abs(best - s['best_accuracy']) < 1e-12
    assert abs(float(rows[-1]['accuracy_reconstructed']) - s['final_accuracy']) < 1e-12
    assert s['eval/accuracy'] == s['eval/answer_score']
    assert s['eval/accuracy'] <= s['eval/format_score'] <= 1
    assert s['train_step'] == 256
    run = root / 'results/runs' / f'size_{n}'
    matches = re.findall(r'Step (\d+): Accuracy: ([\d.]+), Avg Len: ([\d.]+)', (run/'evaluation_log.txt').read_text())
    assert len(matches) == 33
    for row, (step, acc, length) in zip(rows, matches):
        assert int(row['step']) == int(step)
        assert float(row['accuracy_log_4dp']) == float(acc)
        assert float(row['avg_length_chars_log_1dp']) == float(length)
    print(f'size={n}: 33 evaluation points, final={s["final_accuracy"]:.4%}, best={best:.4%}: OK')
samples = list((root/'results/samples').glob('*.json'))
for path in samples:
    obj = json.loads(path.read_text())
    assert obj['columns'] == ['step','prompt','response','ground_truth','reward','format_reward','answer_reward']
    assert len(obj['data']) == 100
    expected_step = int(path.stem.split('_step_')[1])
    assert all(row[0] == expected_step for row in obj['data'])
for entry in json.loads((root/'results/source_manifest.json').read_text()):
    assert hashlib.sha256((root/entry['snapshot']).read_bytes()).hexdigest() == entry['sha256']
print(f'{len(samples)} sample tables validated (100 rows each).')
print('Source snapshot SHA-256 hashes: OK')
print('All exported result checks passed. No GPU training was run.')
