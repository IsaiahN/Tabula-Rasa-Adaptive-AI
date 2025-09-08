import glob
import json
import os
import time

batch_files = sorted(glob.glob(os.path.join('data', 'inventory_batch_*.json')))
if not batch_files:
    print('No batch inventory files found in data/.')
    raise SystemExit(1)

all_entries = []
for bf in batch_files:
    try:
        with open(bf, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            # support both list payloads and dict with 'entries'
            if isinstance(payload, dict) and 'entries' in payload:
                entries = payload['entries']
            elif isinstance(payload, list):
                entries = payload
            else:
                # try best-effort: assume payload is a dict with file-level summary
                entries = payload.get('files', []) if isinstance(payload, dict) else []
            all_entries.extend(entries)
    except Exception as e:
        print(f'Failed to read {bf}: {e}')

# deduplicate by path
seen = {}
merged = []
for e in all_entries:
    p = e.get('path') or e.get('file') or e.get('filename')
    if not p:
        # fallback: stringify entry
        key = json.dumps(e, sort_keys=True)
        if key in seen:
            continue
        seen[key] = True
        merged.append(e)
    else:
        if p in seen:
            # prefer the later entry if duplicates
            continue
        seen[p] = True
        merged.append(e)

ts = int(time.time())
out_path = os.path.join('data', f'inventory_full_{ts}.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({'generated_at': ts, 'count_batches': len(batch_files), 'entries': merged}, f, indent=2)

# write candidate list
candidates = [e for e in merged if e.get('archive_candidate')]
cand_path = os.path.join('data', f'inventory_candidates_{ts}.json')
with open(cand_path, 'w', encoding='utf-8') as f:
    json.dump({'generated_at': ts, 'candidates': candidates, 'count': len(candidates)}, f, indent=2)

print(f'Merged {len(batch_files)} batch files -> {out_path} (entries={len(merged)})')
print(f'Candidates: {len(candidates)} -> {cand_path}')
# print top 20 candidate file paths
for c in candidates[:20]:
    print('-', c.get('path'), 'avg_score=%s' % c.get('avg_score'), 'total_actions=%s' % c.get('total_actions'))

print('Done.')
