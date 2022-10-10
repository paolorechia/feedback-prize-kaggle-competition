from cmath import isnan
import pandas as pd
import math
from tqdm import tqdm
from utils import break_sentences

train_df = pd.read_csv("/data/feedback-prize/sentence_train.csv")

setfit_model_max_length = 256
minimum_chunk_length = 32

broken_sentences = break_sentences(
    train_df, setfit_model_max_length, minimum_chunk_length
)
