# ChatGPTBlock
A Python package for interacting with OpenAI's chat models through the OpenAI API.

## Table Of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Introduction
The main purpose of the package is to have a simple interface for interacting with OpenAI api.
Both in streamable and non-streamable fashion.
OpenAI API itself does not store the history and can throw and error if history is too long.
The package provides the solution to this by simply counting the length of the history and trimming it when needed.

## Features
1. streamable mode
2. supporting the history of the conversation
3. resetting the history of the conversation
4. adding your custom pre-processing

## Installation

### Install Locally
1. `git clone https://github.com/SkuratovichA/chatgpt_block`
2. `cd chatgpt_block`
2. `pip install -e .`

### Install With PIP
`pip install chatgpt_block`


## Usage

### Creating The Conversation
Here's an example of how to use the ChatGPTBlock class:
There are two options of using the class. Streamable and non-streamable.

```python
from chatgpt_block import ChatGPTBlock

# Initialize the ChatGPTBlock instance
chatgpt_block = ChatGPTBlock(
    system_prompt="You are a helpful assistant.",
    openai_api_token="your_openai_api_key",
    model="gpt-4",
)

# Get a response from the model
response = chatgpt_block("Tell me a joke.")
print(response)
```
    
```python
from chatgpt_block import ChatGPTBlock

chatgpt_block = ChatGPTBlock(
    system_prompt="You are a helpful assistant.",
    openai_api_token="your_openai_api_key",
    model="gpt-4",
    stream=True
)

generator = chatgpt_block("Tell me a joke.")
for token in generator:
    print(token, flush=True, end='')
```

### Resetting The Conversation

```python
chatgpt_block("Tell me a joke.")
print(chatgpt_block.answer)

chatgpt_block.reset()

response = chatgpt_block("Tell me a story.")
print(response)
```

### Custom Preprocessor Function
```python
from chatgpt_block import ChatGPTBlock

def custom_preprocessor(input_text: str, some_dictionary: dict, prologue: str) -> str:
    return f"{prologue}\n{input_text}. {some_dictionary}"

chatgpt_block = ChatGPTBlock(
    system_prompt="You are a helpful assistant.",
    openai_api_token="your_openai_api_key",
    model="gpt-4",
    preprocessor=custom_preprocessor,
)

response = chatgpt_block(
    input_text="please check whether this python dictionary is valid", 
    some_dictionary={"hello": "world"}, prologue=''
)
print(response)
```

### Custom Error Handling Function
```python
from chatgpt_block import ChatGPTBlock

def custom_error_handler():
    print("There was an error when communicating with OpenAI API. Please try again later")

chatgpt_block = ChatGPTBlock(
    system_prompt="You are a helpful assistant.",
    openai_api_token="your_openai_api_key",
    model="gpt-4",
    on_error=custom_error_handler,
)

response = chatgpt_block("tell me a joke")
print(response)
```

### Feeding The Model With The Examples
```python
from chatgpt_block import ChatGPTBlock

examples = [
    ("Please add a vegetable to an animal", 
     "Alpaca + avocado = alpacavocado"),
    ("Now I want something with cats", 
     "Cat + Cappuccino = cappucicatto"),
]

chatgpt_block = ChatGPTBlock(
    system_prompt=
    """You are a helpful assistant with a sense of humor. 
       You know everything about animals, vegetables and how to combine them. 
       You must add animals to vegetables no matter what to create funny words.
    """,
    examples=examples,
    openai_api_token="your_openai_api_key",
    model="gpt-4",
)

response = chatgpt_block("Make up some new word with \"tortoise\"")
print(response)
```

## Contributing
This package is free to any ideas. Just create an issue or a pull request on GitHub.

## License
`chatgpt_block` is released under [MIT License](LICENSE).
