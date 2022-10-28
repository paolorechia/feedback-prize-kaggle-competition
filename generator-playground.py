from datetime import datetime
from data_augmentation import GPTNeoGenerator, T0Generator, GPT2Generator

generator = T0Generator()
while True:
    prompt = input("Prompt: ")
    start = datetime.now()
    print(
        generator.generate(
            prompt,
            max_length=64,
            do_sample=True,
            temperature=0.9,
        )
    )
    elapsed_time_in_seconds = (datetime.now() - start).total_seconds()
    print("Elapsed time: ", elapsed_time_in_seconds)
