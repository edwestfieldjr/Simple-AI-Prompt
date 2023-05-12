#!~/.pyenv/versions/3.11.3/envs/transformers/bin/python

import sys
import subprocess
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
# import tensorflow as tf

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'transformers', 'tensorflow', 'torch'])

# tf.get_logger().setLevel('FATAL')

model_name = "google/flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=200)

while True: 
    line = input("\nAsk question: ")
    if 'q' == line.rstrip():
        break
    
    tokens = tokenizer(line, return_tensors="pt")
    outputs = model.generate(**tokens, generation_config=config)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_text = str(decoded[0])
    print(f"\nAnswer: {decoded_text.capitalize()}{('' if decoded_text[-1] in ['.', '?', '!'] else '.')} \n")


