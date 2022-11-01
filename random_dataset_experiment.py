from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd
from load_data import create_train_test_df
import random
import gzip
import json

from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import (
    MultiBlockRidgeCV,
    predict_multi_block,
    score_multi_block,
    fit_multi_block,
)
from utils import attributes, calculate_rmse_score_single, possible_labels


def load_train_dataset(head=3000):
    filepath = "/data/text-datasets/goodreads_reviews_dedup.json.gz"
    count = 0
    data = []
    with gzip.open(filepath) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            data.append(d)

            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    return data


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

    multi_block: MultiBlockRidgeCV
    attribute: str


class GeneticAlgorithm:
    def __init__(
        self, dataset_context: DatasetContext, population_size=20, n_labels=1000
    ):
        self.population_size = population_size
        self.n_labels = n_labels
        self.population = []
        self.fitness = []
        self.best_individual = None
        self.best_fitness = None
        self.dataset_context = dataset_context
        self.number_of_survivals = 2
        self.number_of_randoms = 2
        self.number_of_cross_over = (
            population_size - self.number_of_survivals - self.number_of_randoms
        )

    def initialize_population(self):
        for _ in range(self.population_size):
            self.population.append(
                [random.random() * 4 + 1 for _ in range(self.n_labels)]
            )

    def evaluate_fitness(self, individual):
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

        score = calculate_rmse_score_single(self.dataset_context.y_test, y_pred[0])

        val_pred = predict_multi_block(
            self.dataset_context.multi_block,
            self.dataset_context.attribute,
            self.dataset_context.val_df,
            self.dataset_context.val_indices,
        )
        val_score = calculate_rmse_score_single(self.dataset_context.y_val, val_pred[0])
        print("Test Score: {} || Val Score: {}".format(score, val_score))
        return score

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
            if random.random() < 0.2:
                individual[i] = random.random() * 4 + 1

    def save_best_individual(self, individual, fitness, epoch):
        with open(
            "best_individuals/individual_{}_{}.json".format(fitness, epoch), "w"
        ) as fout:
            json.dump(individual, fout)

    def run(self, num_generations=100):
        for epoch in range(num_generations):
            self.fitness = []
            for idx, individual in enumerate(self.population):
                score = self.evaluate_fitness(individual)
                self.fitness.append((score, idx))
                self.fitness.sort(key=lambda x: x[0])

            # fitness here is error, so smaller is better
            if self.best_individual is None or self.best_fitness > self.fitness[0][0]:
                self.best_individual = self.population[self.fitness[0][1]].copy()
                self.best_fitness = self.fitness[0][0].copy()

            print("Generation: {} || Best Fitness: {}".format(epoch, self.best_fitness))
            survivals = []

            for i in range(self.number_of_survivals):
                survivals.append(self.population[self.fitness[i][1]])

            self.population = survivals.copy()
            for i in range(self.number_of_cross_over):
                parent1 = random.choice(survivals)
                parent2 = random.choice(survivals)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                self.population.append(child)

            for i in range(self.number_of_randoms):
                self.population.append(
                    [random.random() * 4 + 1 for _ in range(self.n_labels)]
                )
            self.save_best_individual(self.best_individual, self.best_fitness, epoch)

# TODO: write text degradation

def main():
    # Load the model
    full_df, val_df = create_train_test_df(0.1, "full")

    full_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    dataset_size = 1000
    epochs = 5000

    model_info = ModelCatalog.DebertaV3
    multi_block_class = MultiBlockRidgeCV
    multi_block = multi_block_class(
        model=ModelStack(
            [
                model_info,
            ],
        ),
        number_blocks=1,
        labels=attributes,
    )

    X_test = multi_block.encode(full_df["full_text"], cache_type="full-0.9")
    full_df["embeddings_0"] = [np.array(e) for e in X_test]
    test_indices = full_df.index.values

    X_val = multi_block.encode(val_df["full_text"], cache_type="val-0.1")
    val_df["embeddings_0"] = [np.array(e) for e in X_val]
    val_indices = val_df.index.values

    train_dataset = load_train_dataset(head=dataset_size - 1)
    train_df = pd.DataFrame(train_dataset)

    X_train = multi_block.encode(
        train_df["review_text"], cache_type=f"review-dataset-{dataset_size}"
    )

    train_df["embeddings_0"] = [np.array(e) for e in X_train]

    train = train_df.index.values

    for attribute in ["cohesion"]:
        print("Attribute", attribute)
        y_test = np.array(full_df[attribute])
        y_val = np.array(val_df[attribute])
        dataset_context = DatasetContext(
            train_df=train_df,
            test_df=full_df,
            val_df=val_df,
            train_indices=train,
            test_indices=test_indices,
            val_indices=val_indices,
            y_test=y_test,
            y_val=y_val,
            multi_block=multi_block,
            attribute="cohesion",
        )

        genetic = GeneticAlgorithm(
            population_size=20, n_labels=dataset_size, dataset_context=dataset_context
        )
        genetic.initialize_population()
        genetic.run(num_generations=epochs)


if __name__ == "__main__":
    main()
