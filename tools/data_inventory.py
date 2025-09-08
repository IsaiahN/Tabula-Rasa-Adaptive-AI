#!/usr/bin/env python3
import os, json, glob, argparse, time
from datetime import datetime

def parse_file(path):
    info = {
        'path': path,
        'bytes': os.path.getsize(path),
        'modified_time': os.path.getmtime(path),
        'parsed_type': 'unknown',
        'total_episodes': 0,
        'total_actions': 0,
        'avg_actions': 0.0,
        'max_actions': 0,
        'avg_score': None,
        'win_rate': None,
        'archive_candidate': False,
    }
    try:
        with open(path, 'r', encoding='utf-8') as f:
            j = json.load(f)
    except Exception as e:
        info['parse_error'] = str(e)
        return info

    # Heuristics for parsed_type
    name = os.path.basename(path)
    if name.startswith('session_session_') and name.endswith('_final.json'):
        info['parsed_type'] = 'session'
        # collect from games_played
        games = j.get('games_played') or {}
        total_eps = 0
        total_actions = 0
        max_actions = 0
        total_score = 0.0
        scored_eps = 0
        wins = 0
        for g in games.values():
            eps = g.get('episodes') or []
            total_eps += len(eps)
            for e in eps:
                a = e.get('actions_taken', 0) or 0
                total_actions += a
                if a > max_actions:
                    max_actions = a
                # final score
                sc = e.get('final_score')
                if sc is not None:
                    try:
                        total_score += float(sc)
                        scored_eps += 1
                        if (sc) and float(sc) > 0:
                            wins += 1
                    except:
                        pass
        info['total_episodes'] = total_eps
        info['total_actions'] = total_actions
        info['max_actions'] = max_actions
        info['avg_actions'] = (total_actions / total_eps) if total_eps else 0.0
        # whatever top-level provides
        info['avg_score'] = j.get('overall_performance', {}).get('overall_average_score') if isinstance(j.get('overall_performance'), dict) else None
        info['win_rate'] = j.get('overall_performance', {}).get('overall_win_rate') if isinstance(j.get('overall_performance'), dict) else None
        if info['avg_score'] is None and scored_eps:
            info['avg_score'] = total_score / scored_eps
        if info['win_rate'] is None and total_eps:
            info['win_rate'] = (wins / total_eps)

    elif name.startswith('meta_learning_session_'):
        info['parsed_type'] = 'meta'
        # meta files often don't have episodes
        info['avg_score'] = j.get('summary', {}).get('avg_score') or j.get('average_score')
        info['win_rate'] = j.get('summary', {}).get('win_rate') or j.get('win_rate')
    elif name.startswith('action_intelligence_'):
        info['parsed_type'] = 'action_intelligence'
        info['avg_score'] = j.get('last_updated') and 0.0
    elif name.startswith('combined_score_'):
        info['parsed_type'] = 'combined'
        info['avg_score'] = j.get('combined_data', {}).get('avg_score')
        info['total_actions'] = j.get('combined_data', {}).get('total_actions', 0)
    else:
        info['parsed_type'] = 'other'

    # archival heuristic will be applied outside, but compute defaults when missing
    return info

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'data'))
    p.add_argument('--heuristic', default='avg_score==0 and total_actions<10')
    args = p.parse_args()

    # Include top-level data JSONs and the canonical subfolders for sessions and meta-learning
    patterns = [
        os.path.join(args.data_dir, '*.json'),
        os.path.join(args.data_dir, 'sessions', '*.json'),
        os.path.join(args.data_dir, 'meta_learning_sessions', '*.json'),
        os.path.join(args.data_dir, 'meta_learning_data', '*.json')
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    paths = sorted(set(paths))
    selected = paths[args.offset: args.offset + args.limit]

    inventory = []
    archive_candidates = []
    for pth in selected:
        info = parse_file(pth)
        # apply chosen heuristic: avg_score == 0 AND total_actions < 10
        avg_score = info.get('avg_score')
        total_actions = info.get('total_actions', 0)
        try:
            is_candidate = (avg_score is not None and float(avg_score) == 0.0 and int(total_actions) < 10)
        except:
            is_candidate = False
        info['archive_candidate'] = bool(is_candidate)
        inventory.append(info)
        if info['archive_candidate']:
            archive_candidates.append(info)

    ts = int(time.time())
    out_path = os.path.join(args.data_dir, f'inventory_batch_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'generated': ts, 'limit': args.limit, 'offset': args.offset, 'heuristic': args.heuristic, 'entries': inventory}, f, indent=2)

    # print short summary
    print(f'Parsed {len(selected)} files, inventory -> {out_path}')
    print(f'Archive candidates (heuristic={args.heuristic}): {len(archive_candidates)}')
    # show top 10 candidates
    for c in archive_candidates[:10]:
        print('-', c['path'], f"avg_score={c.get('avg_score')} total_actions={c.get('total_actions')}")

    # exit
    
