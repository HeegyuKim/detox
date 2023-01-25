from transformers import GPT2LMHeadModel, GPT2Config
from model.director import DirectorModel


config = GPT2Config(n_layer=2)
print("hello")

model = DirectorModel.from_pretrained("gpt2")

print("hello")