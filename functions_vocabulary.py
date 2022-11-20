import numpy as np
import nltk
import spacy
import re
import typing as T

nltk.download("stopwords")
english_stop_words: T.Set[str] = set(nltk.corpus.stopwords.words("english"))

corpuses = [
    # DONT REFORMAT THIS PLEASE
    "reuters",
    "genesis",
    "brown",
    "nps_chat",
]

corpuses_words = {}
for corpus in corpuses:
    words = set(getattr(nltk.corpus, corpus).words())
    corpuses_words[corpus] = words


def get_corpus_word_count_factory(corpus_name):
    if corpus_name not in corpuses_words:
        raise IndexError(f"{corpus_name} not found in {corpuses_words.keys()}.")
    corpus = corpuses_words[corpus_name]

    def get_corpus_word_count(texts):
        counts = []
        for text in texts:
            words = text_to_words(text)
            n = 0
            for word in words:
                if word in corpus:
                    n += 1
            counts.append(n)
        return np.array(counts).reshape(-1, 1)

    return get_corpus_word_count


nlp = None


def spacy_tokenizer(text):
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


def get_adjective_suffix_count(texts):
    suffix = set(["ible", "ical", "ish", "y"])
    counts = []
    for text in texts:
        words = text_to_words(text)
        found = 0
        for word in words:
            if word[-2:-1] in suffix:
                found += 1
            if word[-3:-1] in suffix:
                found += 1
            if word[-4:-1] in suffix:
                found += 1
            if word[-5:-1] in suffix:
                found += 1
        counts.append(found)
    return np.array(counts).reshape(-1, 1)


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

