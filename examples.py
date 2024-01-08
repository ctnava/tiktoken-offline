# array_tokens = encoder.encode("Hello, World!") 
# [9906, 11, 4435, 0] || 4

# array_tokens = encoder.encode("I don't feel so good...")
# 40, 1541, 956, 2733, 779, 1695, 1131] || 7

# array_tokens = encoder.encode("The FitnessGram Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues.")
# [791, 36808, 65325, 393, 9779, 3475, 374, 264, 2814, 380, 425, 91490, 8824, 1296, 430, 72859, 5334, 810, 5107, 439, 433, 9731, 13] || 23

import tiktoken_offline as tiktoken

available_model_names = tiktoken.list_model_names()
encoder = tiktoken.encoding_for_model("gpt-4")
array_tokens = encoder.encode("Hello, World!") 
num_tokens = len(array_tokens)

available_encoding_names = tiktoken.list_encoding_names()
encoder = tiktoken.get_encoding("cl100k_base")
array_tokens = encoder.encode("Hello, World!") 
num_tokens = len(array_tokens)

print(available_model_names)
print(available_encoding_names)
print(encoder)
print(array_tokens)
print(num_tokens)