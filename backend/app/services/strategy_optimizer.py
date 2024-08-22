import numpy as np
from typing import Dict
from sqlalchemy.orm import Session
from deap import base, creator, tools, algorithms
from app.services.data_collection import DataCollector

class StrategyOptimizer:
    def __init__(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=4)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_strategy)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_strategy(self, individual):
        # В реальном приложении здесь будет оценка стратегии на исторических данных
        # Для примера используем простую функцию
        return (sum(individual),)

    async def optimize(self, params: Dict, db: Session) -> Dict:
        # Получаем исторические данные
        data_collector = DataCollector()
        historical_data = await data_collector.get_historical_data(
            start_date=params.get('start_date'),
            end_date=params.get('end_date'),
            symbol=params.get('symbol', 'BTC/USDT'),
            timeframe=params.get('timeframe', '1d')
        )

        # Используйте historical_data в вашей логике оптимизации
        # ...

        pop = self.toolbox.population(n=params['populationSize'])
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2,
                                           ngen=params['generations'], stats=stats,
                                           halloffame=hof, verbose=True)

        return {
            'bestFitness': hof[0].fitness.values[0],
            'bestParameters': {
                'param1': hof[0][0],
                'param2': hof[0][1],
                'param3': hof[0][2],
                'param4': hof[0][3]
            }
        }