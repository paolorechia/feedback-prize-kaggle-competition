from transformers import pipeline
from datetime import datetime

generator = pipeline(
    "text-generation",
    model="/data/dropout_test/EleutherAI/gpt-neo-1.3B",
    device="cuda:0",
)

while True:
    prompt = input("Prompt: ")
    start = datetime.now()
    print(
        generator(
            prompt,
            max_length=1024,
            do_sample=True,
            top_p=0.95,
            top_k=60,
            temperature=0.9,
            num_return_sequences=1,
        )
    )
    elapsed_time_in_seconds = (datetime.now() - start).total_seconds()
    print("Elapsed time: ", elapsed_time_in_seconds)
