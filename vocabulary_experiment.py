import load_data
import utils
import numpy as np
import nltk
import spacy
import re
import sklearn as sk
import typing as T
import tqdm
import json
import os

nltk.download("stopwords")
english_stop_words: T.List[str] = nltk.corpus.stopwords.words("english")
nlp = None


def _spacy_tokenizer(text):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append(
            {
                "text": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "lemma": token.lemma_,
                "dep": token.dep_,
                "n_children": len([child for child in token.children]),
                "morph_case": token.morph.get("Case"),
                "morph_person": token.morph.get("Person"),
                "morph_verb_form": token.morph.get("VerbForm"),
            }
        )
    return tokens


def _custom_tokenizer(text):
    dedup_spaces = re.sub("\s+", " ", text)
    lowered = dedup_spaces.lower()
    return lowered.split(" ")


def text_to_words(text, tokenizer=_custom_tokenizer):
    return tokenizer(text)


def get_stop_word_count(texts):
    word_count = []
    for text in texts:
        words = text_to_words(text)
        num_stop_words = 0
        for word in words:
            if word in english_stop_words:
                num_stop_words += 1
        word_count.append(num_stop_words)

    return np.array(word_count).reshape(-1, 1)


def get_word_count_feature(texts):
    word_count = []
    for text in texts:
        words = text_to_words(text)
        word_count.append(len(set(words)))
    return np.array(word_count).reshape(-1, 1)


def get_lexical_diversity(texts):
    lexical_diversity = []
    for text in texts:
        words = text_to_words(text)
        lexical_diversity.append(len(set(words)) / len(words))
    return np.array(lexical_diversity).reshape(-1, 1)


def get_text_length(texts):
    text_lengths = []
    for text in texts:
        text_lengths.append(len(text))
    return np.array(text_lengths).reshape(-1, 1)


def get_prop_text_length(texts):
    text_lengths = []
    total_length = 0
    for text in texts:
        len_ = len(text)
        total_length += len_
        text_lengths.append(len(text))
    prop_text_lengths = [tl / total_length for tl in text_lengths]
    return np.array(prop_text_lengths).reshape(-1, 1)


def get_bigram_count(texts):
    bigram_count = []
    for text in texts:
        words = text_to_words(text)
        bigrams = list(nltk.bigrams(words))
        bigram_count.append(len(bigrams))
    return np.array(bigram_count).reshape(-1, 1)


def max_word_length(texts):
    max_word_lengths = []
    for text in texts:
        words = text_to_words(text)
        lengths = [len(word) for word in words]
        max_word_lengths.append(max(lengths))
    return np.array(max_word_lengths).reshape(-1, 1)


def avg_word_length(texts):
    avg_word_lengths = []
    for text in texts:
        words = text_to_words(text)
        lengths = [len(word) for word in words]
        avg_word_lengths.append(sum(lengths) / len(lengths))
    return np.array(avg_word_lengths).reshape(-1, 1)


def big_word_count(texts):
    big_word_threshold = 10
    big_words = []
    for text in texts:
        words = text_to_words(text)
        counts = len([len(word) for word in words if len(word) > big_word_threshold])
        big_words.append(counts)
    return np.array(big_words).reshape(-1, 1)


def small_word_count(texts):
    small_word_threshold = 4
    small_words = []
    for text in texts:
        words = text_to_words(text)
        counts = len([len(word) for word in words if len(word) < small_word_threshold])
        small_words.append(counts)
    return np.array(small_words).reshape(-1, 1)


def get_vogal_count(texts):
    letters = "aeiou"
    counts = []
    for text in texts:
        words = text_to_words(text)
        text_ = " ".join(words)
        n_letters = 0
        for char in text_:
            if char in letters:
                n_letters += 1
        counts.append(n_letters)
    return np.array(counts).reshape(-1, 1)


def get_consoant_count(texts):
    letters = "bcdefghjklmnpqrstvxywz"
    counts = []
    for text in texts:
        words = text_to_words(text)
        text_ = " ".join(words)
        n_letters = 0
        for char in text_:
            if char in letters:
                n_letters += 1
        counts.append(n_letters)
    return np.array(counts).reshape(-1, 1)


def get_vogal_ratio(texts):
    letters = "aeiou"
    counts = []
    for text in texts:
        words = text_to_words(text)
        text_ = " ".join(words)
        n_letters = 0
        for char in text_:
            if char in letters:
                n_letters += 1
        counts.append(len(text_) / n_letters)
    return np.array(counts).reshape(-1, 1)


def get_verb_count(spacy_tokens):
    verbs_count = []
    for tokens in spacy_tokens:
        verbs = 0
        for token in tokens:
            if token["pos"] == "VERB":
                verbs += 1
        verbs_count.append(verbs)
    return np.array(verbs_count).reshape(-1, 1)


def get_unique_verb_count(spacy_tokens):
    verbs_count = []
    for tokens in spacy_tokens:
        verb_set = set()
        for token in tokens:
            if token["pos"] == "VERB":
                verb_set.add(token["text"])
        verbs_count.append(len(verb_set))
    return np.array(verbs_count).reshape(-1, 1)


def get_verb_ratio(spacy_tokens):
    verbs_count = []
    for tokens in spacy_tokens:
        verbs = 0
        for token in tokens:
            if token["pos"] == "VERB":
                verbs += 1
        if verbs == 0:
            print("No verbs!!")
            verbs_count.append(0)
        else:
            verbs_count.append(len(tokens) / verbs)
    return np.array(verbs_count).reshape(-1, 1)


def get_verb_morph_count(spacy_tokens):
    verbs_count = []
    for tokens in spacy_tokens:
        verbs = 0
        for token in tokens:
            if token["pos"] == "VERB":
                if token["morph_verb_form"] == "Ger":
                    verbs += 1
        verbs_count.append(verbs)
    return np.array(verbs_count).reshape(-1, 1)


def get_lemma_count(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        lemmas = []
        for token in tokens:
            lemmas.append(token["lemma"])
        nouns_count.append(len(set(lemmas)))
    return np.array(nouns_count).reshape(-1, 1)


def get_bigram_lemma_count(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        lemmas = []
        for token in tokens:
            lemmas.append(token["lemma"])
        unique_bigrams = set(list(nltk.bigrams(lemmas)))
        nouns_count.append(len(unique_bigrams))
    return np.array(nouns_count).reshape(-1, 1)


def get_tag_count(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        tags = []
        for token in tokens:
            tags.append(token["tag"])
        nouns_count.append(len(set(tags)))
    return np.array(nouns_count).reshape(-1, 1)


def get_noun_count(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        nouns = 0
        for token in tokens:
            if token["pos"] == "NOUN":
                nouns += 1
        nouns_count.append(nouns)
    return np.array(nouns_count).reshape(-1, 1)


def get_noun_ratio(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        nouns = 0
        for token in tokens:
            if token["pos"] == "NOUN":
                nouns += 1
        if nouns == 0:
            print("No nouns!!")
            nouns_count.append(0)
        else:
            nouns_count.append(len(tokens) / nouns)
    return np.array(nouns_count).reshape(-1, 1)


def get_noun_third_person_count(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        nouns = 0
        for token in tokens:
            if token["pos"] == "NOUN" and token["morph_person"] == "3":
                nouns += 1
        nouns_count.append(nouns)
    return np.array(nouns_count).reshape(-1, 1)


def get_avg_num_children(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        n_children = []
        for token in tokens:
            n_children.append(token["n_children"])
        nouns_count.append(sum(n_children) / len(n_children))
    return np.array(nouns_count).reshape(-1, 1)


def get_max_num_children(spacy_tokens):
    nouns_count = []
    for tokens in spacy_tokens:
        n_children = []
        for token in tokens:
            n_children.append(token["n_children"])
        nouns_count.append(max(n_children))
    return np.array(nouns_count).reshape(-1, 1)


used_features_functions = [
    # get_vogal_ratio,
    get_stop_word_count,
    get_text_length,
    get_lexical_diversity,
    get_word_count_feature,
    get_bigram_count,
    max_word_length,
    avg_word_length,
    big_word_count,
    small_word_count,
    get_vogal_count,
    get_consoant_count,
]

used_spacy_features = [
    # get_unique_verb_count,
    # get_noun_ratio,
    # get_verb_morph_count,
    get_verb_count,
    get_noun_third_person_count,
    get_verb_ratio,
    get_noun_count,
    get_lemma_count,
    get_tag_count,
    get_bigram_lemma_count,
    get_avg_num_children,
    get_max_num_children,
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
            X_spacy_tokens.append(_spacy_tokenizer(text))

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

