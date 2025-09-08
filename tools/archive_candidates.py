import os, json, time, shutil, glob

cand_files = sorted(glob.glob(os.path.join('data', 'inventory_candidates_*.json')))
if not cand_files:
    print('No inventory_candidates_*.json found in data/. Run merge_inventories.py first or generate candidate file.')
    raise SystemExit(1)

# take the latest
latest = cand_files[-1]
with open(latest, 'r', encoding='utf-8') as f:
    data = json.load(f)
    candidates = data.get('candidates', [])

if not candidates:
    print('No candidates found in', latest)
    raise SystemExit(0)

ts = time.strftime('%Y%m%dT%H%M%S')
archive_dir = os.path.join('data', 'archive_non_improving', ts)
os.makedirs(archive_dir, exist_ok=True)

manifest = []
for c in candidates:
    p = c.get('path')
    if not p:
        continue
    fname = os.path.basename(p)
    dest = os.path.join(archive_dir, fname)
    try:
        shutil.move(p, dest)
        manifest.append({'from': p, 'to': dest, 'reason': c.get('reason', 'avg_score==0 and total_actions<10')})
    except Exception as e:
        print('Failed moving', p, '->', dest, e)

man_path = os.path.join(archive_dir, 'manifest.json')
with open(man_path, 'w', encoding='utf-8') as f:
    json.dump({'moved_at': ts, 'count': len(manifest), 'manifest': manifest}, f, indent=2)

print(f'Moved {len(manifest)} files to {archive_dir}')
print('Manifest at', man_path)
