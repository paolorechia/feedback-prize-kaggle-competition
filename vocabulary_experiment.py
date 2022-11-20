import load_data
import utils
import numpy as np
import sklearn as sk
import tqdm
import json
import os

import functions_vocabulary as fv


used_features_functions = [
    # get_vogal_ratio,
    fv.get_stop_word_count,
    fv.get_text_length,
    fv.get_lexical_diversity,
    fv.get_word_count_feature,
    fv.get_bigram_count,
    fv.max_word_length,
    fv.avg_word_length,
    fv.big_word_count,
    fv.small_word_count,
    fv.get_vogal_count,
    fv.get_consoant_count,
    fv.get_adjective_suffix_count,
]
for corpus in fv.corpuses:
    used_features_functions.append(fv.get_corpus_word_count_factory(corpus))
    used_features_functions.append(fv.get_corpus_bigram_count_factory(corpus))

used_spacy_features = [
    # get_unique_verb_count,
    # get_noun_ratio,
    # get_verb_morph_count,
    fv.get_verb_count,
    fv.get_noun_third_person_count,
    fv.get_verb_ratio,
    fv.get_noun_count,
    fv.get_lemma_count,
    fv.get_tag_count,
    fv.get_bigram_lemma_count,
    fv.get_avg_num_children,
    fv.get_max_num_children,
]


def generate_features(texts, feature_functions):
    features = None
    for f in feature_functions:
        if features is None:
            features = f(texts)
        else:
            features = np.hstack([features, f(texts)])
    return features


def generate_spacy_tokens(texts, type="train"):
    X_spacy_tokens_fp = f"x_{type}_spacy_tokens.json"
    if os.path.exists(X_spacy_tokens_fp):
        with open(X_spacy_tokens_fp) as fp:
            X_spacy_tokens = json.load(fp)
    else:
        X_spacy_tokens = []
        for text in tqdm.tqdm(iterable=texts, total=len(texts)):
            X_spacy_tokens.append(fv.spacy_tokenizer(text))

        with open(X_spacy_tokens_fp, "w") as fp:
            json.dump(X_spacy_tokens, fp)
    return X_spacy_tokens


train_df, test_df = load_data.create_train_test_df(test_size=0.2, dataset="full")

train_text = train_df["full_text"]
y_train = train_df["vocabulary"]

print("Generating train features")
X_train = generate_features(train_text, used_features_functions)

X_train_spacy_tokens = generate_spacy_tokens(train_text, "train")
if used_spacy_features:
    X_train_spacy_features = generate_features(
        X_train_spacy_tokens, used_spacy_features
    )
    X_train = np.hstack([X_train, X_train_spacy_features])

test_text = test_df["full_text"]
y_test = test_df["vocabulary"]

print("Generating test features")
X_test = generate_features(test_text, used_features_functions)
X_test_spacy_tokens = generate_spacy_tokens(test_text, "test")

if used_spacy_features:
    X_test_spacy_features = generate_features(X_test_spacy_tokens, used_spacy_features)
    X_test = np.hstack([X_test, X_test_spacy_features])

ridge = sk.linear_model.RidgeCV()

print("Fitting")
ridge.fit(X_train, y_train)
preds = ridge.predict(X_test)
score = utils.calculate_rmse_score_single(y_test, preds)
print("Score on vocabulary: ", score)

