from dataclasses import InitVar, dataclass
from typing import Any
import numpy as np
import pandas as pd
from load_data import create_train_test_df
import random
import gzip
import json
import string

from model_catalog import ModelCatalog
from model_stacker import ModelStack
from pre_trained_st_model import (
    MultiBlockRidgeCV,
    MultiBlockSGD,
    MultiBlockRidge,
    MultiBlockBayensianRidge,
    predict_multi_block,
    fit_multi_block,
)
from utils import (
    attributes,
    calculate_rmse_score_single,
    fit_float_score_to_nearest_valid_point,
    possible_labels,
)
import re

import warnings

warnings.filterwarnings("ignore")


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

    multi_block: Any
    attribute: str


class GeneticAlgorithm:
    def __init__(
        self,
        dataset_context: DatasetContext,
        population_size=20,
        n_labels=1000,
        first_ancestor=None,
    ):
        self.dataset_context = dataset_context
        self.population_size = population_size
        self.n_labels = n_labels
        self.first_ancestor = first_ancestor

        self.population = []

        if first_ancestor is not None:
            self.population.append(first_ancestor)

        self.best_individual = None
        self.best_fitness = None

        self.number_of_survivals = int(self.population_size * 0.2) - 1
        self.number_of_randoms = int(self.population_size * 0.1)
        self.number_of_cross_over = (
            population_size - self.number_of_survivals - self.number_of_randoms
        )

    def initialize_population(self):
        for _ in range(self.population_size):
            self.population.append(
                [
                    fit_float_score_to_nearest_valid_point(random.random() * 4 + 1)
                    for _ in range(self.n_labels)
                ]
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
                individual[i] = fit_float_score_to_nearest_valid_point(
                    random.random() * 4 + 1
                )

    def save_best_individual(self, individual, fitness, epoch):
        with open(
            "best_individuals/individual_{}_{}.json".format(fitness, epoch), "w"
        ) as fout:
            ind = [fit_float_score_to_nearest_valid_point(x) for x in individual]
            # print(ind)
            json.dump(ind, fout)

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
                    [
                        fit_float_score_to_nearest_valid_point(random.random()) * 4 + 1
                        for _ in range(self.n_labels)
                    ]
                )
            if epoch % (num_generations // 10) == 0:
                print(
                    "Generation: {} || Best Fitness: {}".format(
                        epoch, self.best_fitness
                    )
                )
                self.evaluate_fitness(self.best_individual, evaluate_val=True)
                print("------------------------\n\n")
                self.save_best_individual(
                    self.best_individual, self.best_fitness, epoch
                )


def _degradate_by_random_typos(text, p=0.1):
    text = list(text)
    for i in range(len(text)):
        if random.random() < p:
            text[i] = random.choice(string.ascii_lowercase)
    return "".join(text)


def _degradate_by_deleting_random_word(text, p=0.1):
    text = text.split()
    for i in range(len(text)):
        if random.random() < p:
            text[i] = ""
    return " ".join(text)


def _degradate_by_deleting_random_char(text, p=0.1):
    text = list(text)
    for i in range(len(text)):
        if random.random() < p:
            text[i] = ""
    return "".join(text)


def _degradate_by_changing_word_order(text):
    text = text.split()
    if len(text) <= 1:
        return text

    number_of_swapped_words = random.randint(1, len(text) // 2)
    for _ in range(number_of_swapped_words):
        i = random.randint(0, len(text) - 1)
        j = random.randint(0, len(text) - 1)
        text[i], text[j] = text[j], text[i]
    return " ".join(text)


def degradate_text(text):
    if random.random() < 0.25:
        text = _degradate_by_random_typos(text)
    if random.random() < 0.25:
        text = _degradate_by_deleting_random_word(text)
    if random.random() < 0.25:
        text = _degradate_by_deleting_random_char(text)
    if random.random() < 0.25:
        text = _degradate_by_changing_word_order(text)
    return text


def degradate_df_text(
    df: pd.DataFrame,
    text_label: str,
    degradation_rate: float,
    minimum_text_length: int = 64,
):
    new_df = pd.DataFrame()
    texts = df[text_label]
    new_texts = []
    for text in texts:
        if random.random() > degradation_rate:
            new_texts.append(text)
        else:
            degradated_text = degradate_text(text)
            if len(degradated_text) <= minimum_text_length:
                continue
            new_texts.append(degradated_text)
    new_df[text_label] = new_texts
    return new_df


def remove_repeated_whitespaces(text):
    return re.sub(r"\s+", " ", text)


def main():
    # Load the model
    train_size = 0.2
    val_size = 1 - train_size
    test_df, val_df = create_train_test_df(train_size, "full")

    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    dataset_size = 100000
    epochs = 10000
    degradation_rate = 0.9
    population_size = 10

    model_info = ModelCatalog.DebertaV3
    multi_block_class = MultiBlockRidge
    multi_block = multi_block_class(
        model=ModelStack(
            [
                model_info,
            ],
        ),
        number_blocks=1,
        labels=attributes,
    )
    test_df["full_text"] = test_df["full_text"].apply(remove_repeated_whitespaces)
    X_test = multi_block.encode(
        test_df["full_text"], cache_type=f"no-ws-full-{train_size}"
    )
    test_df["embeddings_0"] = [np.array(e) for e in X_test]
    test_indices = test_df.index.values

    val_df["full_text"] = val_df["full_text"].apply(remove_repeated_whitespaces)
    X_val = multi_block.encode(val_df["full_text"], cache_type=f"no-ws-val-{val_size}")
    val_df["embeddings_0"] = [np.array(e) for e in X_val]
    val_indices = val_df.index.values

    train_dataset = load_train_dataset(head=dataset_size - 1)
    train_df = pd.DataFrame(train_dataset)

    train_df["review_text"] = train_df["review_text"].apply(remove_repeated_whitespaces)
    train_df = degradate_df_text(
        train_df, "review_text", degradation_rate=degradation_rate
    )
    print(train_df.review_text.head())

    X_train = multi_block.encode(
        train_df["review_text"],
        cache_type=f"no-ws-degraded-v4-{degradation_rate}-review-dataset-{dataset_size}",
    )

    train_df["embeddings_0"] = [np.array(e) for e in X_train]

    train_indices = train_df.index.values

    initial_fit_df = test_df

    for attribute in ["cohesion"]:
        print("Attribute", attribute)
        y_test = np.array(test_df[attribute])
        y_val = np.array(val_df[attribute])
        y_initial = np.array(initial_fit_df[attribute])
        y_initials = [y_initial]

        fit_multi_block(
            multi_block,
            attribute,
            initial_fit_df,
            initial_fit_df.index.values,
            y_initial,
            y_initials,
        )

        first_pred = predict_multi_block(
            multi_block, attribute, train_df, train_indices
        )
        first_pred = first_pred[0]
        fit_multi_block(
            multi_block,
            attribute,
            train_df,
            train_indices,
            first_pred,
            [first_pred],
        )
        second_pred = predict_multi_block(multi_block, attribute, test_df, test_indices)
        second_pred = second_pred[0]
        second_pred = [fit_float_score_to_nearest_valid_point(x) for x in second_pred]

        initial_score = calculate_rmse_score_single(y_test, second_pred)
        print("Initial score: ", initial_score)

        dataset_context = DatasetContext(
            train_df=train_df,
            test_df=test_df,
            val_df=val_df,
            train_indices=train_indices,
            test_indices=test_indices,
            val_indices=val_indices,
            y_test=y_test,
            y_val=y_val,
            multi_block=multi_block,
            attribute="cohesion",
        )

        genetic = GeneticAlgorithm(
            population_size=population_size,
            n_labels=len(train_df),
            dataset_context=dataset_context,
            first_ancestor=first_pred,
        )
        genetic.initialize_population()
        genetic.run(num_generations=epochs)
        train_df[attribute] = [
            fit_float_score_to_nearest_valid_point(x) for x in genetic.best_individual
        ]
        train_df.rename(columns={"review_text": "full_text"}, inplace=True)
        train_df["text_id"] = train_df.index.values
        print("Best fitness", genetic.best_fitness)
        train_df.drop(columns=["embeddings_0"]).to_csv(
            f"./best_fit_{attribute}_score_{genetic.best_fitness}.csv"
        )


if __name__ == "__main__":
    main()
