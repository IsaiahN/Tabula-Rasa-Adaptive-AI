import glob, os, subprocess, math, time

files = sorted(glob.glob(os.path.join('data', '*.json')))
total = len(files)
if total == 0:
    print('No data/*.json files found')
    raise SystemExit(1)

batch_size = 100
batches = math.ceil(total / batch_size)
print(f'Total files: {total}, batches of {batch_size}: {batches}')

# We already ran offset 0 earlier; start from 100
start = 100
for offset in range(start, total, batch_size):
    print('\n--- Running batch offset', offset, 'limit', batch_size)
    cmd = ['python', os.path.join('tools', 'data_inventory.py'), '--limit', str(batch_size), '--offset', str(offset)]
    try:
        r = subprocess.run(cmd, check=False)
        print('Return code', r.returncode)
        if r.returncode != 0:
            print('Non-zero return code for offset', offset, ' — continuing')
    except Exception as e:
        print('Exception running batch', offset, e)
    # small sleep to avoid hammering disk
    time.sleep(0.2)

print('\nAll batches requested. Note: if offsets changed during run, check data/inventory_batch_*.json')
