Running TB-Seed minimal mode

This README explains the minimal mode runner used for quick experiments and CI smoke tests.

Usage (developer):

- Run with real API manager (if configured):

```powershell
python scripts\run_minimal.py --game-id mygame --max-actions 50
```

- Run in CI/test mode using the stub ARC3 API (explicit opt-in flag required):

```powershell
python scripts\run_minimal.py --use-stub-api-for-ci --game-id test_game --max-actions 2
```

Notes:
- Per the seed contract, production/research runs must use a real ARC3 API and DB-backed persistence.
- The stub API is allowed only for CI/test runs and must be explicitly opted-in with `--use-stub-api-for-ci` or the environment variable `USE_STUB_API_FOR_CI=1`.
- Tests live in `/tests`. The CI workflow runs those tests and a minimal smoke invocation.
