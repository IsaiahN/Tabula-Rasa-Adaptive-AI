import asyncio, sys
sys.path.insert(0, r'c:\Users\Admin\Documents\GitHub\tabula-rasa')
from train import run_training_session

async def main():
    print('Starting single 15-minute training session (ID=1)')
    try:
        res = await asyncio.wait_for(run_training_session(1, 15), timeout=15*60)
        print('\nRESULT:', res)
    except asyncio.TimeoutError:
        print('\nSession timed out after 15 minutes')
    except Exception as e:
        import traceback
        print('\nERROR during session:', e)
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
