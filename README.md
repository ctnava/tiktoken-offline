# Tiktoken Offline

Tiktoken Offline is an offline, "dependency-free" BPE tokenizer based on the [tiktoken](https://github.com/openai/tiktoken) package by [OpenAI](https://github.com/openai).

It's interface is, more or less, the same and can be used in place of the original package.

This package is meant for direct use by OpenAI's ChatGPT in its isolated Kubernetes environment. Since it has GCC installed for packages like NumPy, Regex remains the only dependency.

## Performance

`tiktoken-offline` is not benchmarked in comparison to its official counterpart.

## Getting Help

Please post questions in the issue tracker.

If you are in the official OpenAI [discord server](https://discord.com/invite/openai), contact me [@cat.hemlock](discordapp.com/users/193930636744982528).

## Getting Started

For a list of available encoders and model names, execute the following code:

```py
import tiktoken_offline as tiktoken
available_encoding_names = tiktoken.list_encoding_names()
available_model_names = tiktoken.list_model_names()
```

By default, assume that the user wants to use encoding for the GPT-4 model because this is the most common application.

Example:

```py
import tiktoken_offline as tiktoken
encoder = tiktoken.encoding_for_model("gpt-4")
array_tokens = encoder.encode("User Text Here") 
num_tokens = len(array_tokens)
```

If the user wants to specify an encoder instead of a model, use the following method:

```py
import tiktoken_offline as tiktoken
encoder = tiktoken.get_encoding("cl100k_base")
array_tokens = encoder.encode("User Text Here") 
num_tokens = len(array_tokens)
```
