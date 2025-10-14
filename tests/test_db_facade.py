import os
import tempfile
import time
from src.database.db_facade import DBFacade


def test_db_facade_upsert_and_end_session():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db = DBFacade(db_path=path)
        sid = 'test-session-1'
        start = time.strftime('%Y-%m-%dT%H:%M:%S')
        db.upsert_session(sid, start)
        # end session
        end = time.strftime('%Y-%m-%dT%H:%M:%S')
        db.end_session(sid, end)
        # add a game and action
        gid = 'test-game-1'
        db.upsert_game(gid, None, None, start)
        aid = db.add_action(sid, gid, {'action': 'noop'}, {'result': 'ok'}, end)
        assert isinstance(aid, int)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
