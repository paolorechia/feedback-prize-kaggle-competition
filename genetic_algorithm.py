from dataclasses import dataclass
import pandas as pd
import numpy as np
import random
import json
from typing import Any
from pre_trained_st_model import fit_multi_block, score_multi_block, predict_multi_block
from utils import (
    calculate_rmse_score_single,
)


@dataclass
class DatasetContext:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: pd.DataFrame

    train_indices: np.ndarray
    test_indices: np.ndarray
    val_indices: np.ndarray

    y_test: np.ndarray
    y_val: np.ndarray

    multi_block: Any
    attribute: str


class GeneticAlgorithm:
    def __init__(
        self,
        dataset_context: DatasetContext,
        population_size=20,
        n_labels=1000,
        first_ancestor=None,
        cache_key=None,
    ):
        self.dataset_context = dataset_context
        self.population_size = population_size
        self.n_labels = n_labels
        self.first_ancestor = first_ancestor
        self.cache_key = cache_key

        self.population = []

        if first_ancestor is not None:
            self.population.append(first_ancestor)

        if cache_key is not None:
            first_ancestor = self.load_first_ancestor(cache_key)
            if first_ancestor is not None:
                print("Loaded first ancestor")
                self.population.append(first_ancestor)

        self.best_individual = None
        self.best_fitness = None

        self.number_of_survivals = int(self.population_size * 0.8) - 1
        self.number_of_randoms = int(self.population_size * 0.1)
        self.number_of_cross_over = population_size - self.number_of_survivals

    def initialize_population(self):
        for _ in range(self.population_size):
            self.population.append(
                [random.random() * 4 + 1 for _ in range(self.n_labels)]
            )

    def evaluate_fitness(self, individual, evaluate_val=False):
        y_trains = [individual for _ in range(1)]

        fit_multi_block(
            self.dataset_context.multi_block,
            self.dataset_context.attribute,
            self.dataset_context.train_df,
            self.dataset_context.train_indices,
            individual,
            y_trains,
        )
        # # Predict
        y_pred = predict_multi_block(
            self.dataset_context.multi_block,
            self.dataset_context.attribute,
            self.dataset_context.test_df,
            self.dataset_context.test_indices,
        )
        y_pred = y_pred[0]
        score = calculate_rmse_score_single(self.dataset_context.y_test, y_pred)

        val_score = None
        if evaluate_val:
            val_pred = predict_multi_block(
                self.dataset_context.multi_block,
                self.dataset_context.attribute,
                self.dataset_context.val_df,
                self.dataset_context.val_indices,
            )
            val_score = calculate_rmse_score_single(
                self.dataset_context.y_val, val_pred[0]
            )
            print("Test Score: {} || Val Score: {}".format(score, val_score))
        else:
            print("Test Score: {}".format(score))
        return score, val_score

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.n_labels):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def mutate(self, individual):
        for i in range(self.n_labels):
            if random.random() < 0.05:
                individual[i] = random.random() * 4 + 1

    def save_best_individual(self, individual, fitness, val_score, epoch):
        with open(
            "best_individuals/{}_fitness_{}_val_score_{}_epoch_{}.json".format(
                self.cache_key, fitness, val_score, epoch
            ),
            "w",
        ) as fout:
            json.dump(individual, fout)

        with open(
            "best_individuals/{}_best_individual.json".format(self.cache_key),
            "w",
        ) as fout:
            json.dump(individual, fout)

    def load_first_ancestor(self, cache_key):
        try:
            with open(
                "best_individuals/{}_best_individual.json".format(cache_key), "r"
            ) as fin:
                return json.load(fin)
        except FileNotFoundError:
            return None

    def run(self, num_generations=100):
        for epoch in range(num_generations + 1):
            self.fitness = []
            for idx, individual in enumerate(self.population):
                score, _ = self.evaluate_fitness(individual)
                self.fitness.append((score, idx))
                self.fitness.sort(key=lambda x: x[0])

            # fitness here is error, so smaller is better
            if self.best_individual is None or self.best_fitness > self.fitness[0][0]:
                self.best_individual = self.population[self.fitness[0][1]].copy()
                self.best_fitness = self.fitness[0][0].copy()

            survivals = []

            for i in range(self.number_of_survivals):
                survivals.append(self.population[self.fitness[i][1]])

            self.population = survivals.copy()
            options = survivals
            for i in range(self.number_of_randoms):
                options.append([random.random() * 4 + 1 for _ in range(self.n_labels)])

            for i in range(self.number_of_cross_over):
                parent1 = random.choice(survivals)
                parent2 = random.choice(options)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                self.population.append(child)

            if epoch % (num_generations // 10) == 0:
                print(
                    "Generation: {} || Best Fitness: {}".format(
                        epoch, self.best_fitness
                    )
                )
                score, val_score = self.evaluate_fitness(
                    self.best_individual, evaluate_val=True
                )
                print("------------------------\n\n")
                self.save_best_individual(
                    self.best_individual, self.best_fitness, val_score, epoch
                )
