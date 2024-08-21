import json
import os

class StrategyStorage:
    def __init__(self, storage_dir='strategy_storage'):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def save_strategy(self, strategy_name, params):
        file_path = os.path.join(self.storage_dir, f"{strategy_name}.json")
        with open(file_path, 'w') as f:
            json.dump(params, f)

    def load_strategy(self, strategy_name):
        file_path = os.path.join(self.storage_dir, f"{strategy_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    def list_strategies(self):
        return [f.split('.')[0] for f in os.listdir(self.storage_dir) if f.endswith('.json')]
