import asyncio, json

async def main():
    try:
        from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
        cl = ContinuousLearningLoop(arc_agents_path='.', tabula_rasa_path='.')
    except Exception as e:
        print(json.dumps({'status': 'init_failed', 'reason': str(e)}))
        return

    try:
        res = await cl.verify_api_connection()
        # Print a reduced, non-sensitive summary
        summary = {
            'api_accessible': bool(res.get('api_accessible')),
            'total_games_available': int(res.get('total_games_available', 0)),
            'arc_agents_present': bool(res.get('arc_agents_available')),
            'connection_timestamp': res.get('connection_timestamp')
        }
        print(json.dumps({'status': 'ok', 'summary': summary}))
    except Exception as e:
        print(json.dumps({'status': 'failed', 'reason': str(e)}))

if __name__ == '__main__':
    asyncio.run(main())
