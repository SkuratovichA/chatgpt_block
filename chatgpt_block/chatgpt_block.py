"""
This module contains the ChatGPTBlock class for interacting with OpenAI's GPT-4 model through the API.
"""

from types import GeneratorType
from typing import Union, List, Tuple, Any, Generator, Optional
import openai
import tiktoken
from datetime import datetime
import os
import logging
import requests
from openai.openai_object import OpenAIObject

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '[%(asctime)s - %(funcName)12s() ] >>> %(message)s',
    '%H:%M'
))
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


def compute_time_elapsed(start, end):
    time_elapsed = end - start
    time_elapsed = time_elapsed.seconds + time_elapsed.microseconds / 10.0 ** 6
    return time_elapsed


class SimpleStringIterator:
    def __init__(self, string):
        self.generator = (f'{tok} ' for tok in string.split())

    def __iter__(self):
        return self.generator

    def __next__(self):
        return next(self.generator)


class ChatGPTBlock:
    """
    A class for interacting with OpenAI's chat model through the API.
    """
    def __init__(
            self,
            system_prompt: str,
            examples: Union[None, List[Tuple[Union[str, dict], Union[str, dict]]]] = None,
            openai_api_token: Union[str, None] = None,
            model: str = 'gpt-4',
            max_output_length: int = 400,
            stream: bool = False,
            temperature=0.001,  # high temperature makes the output more unstable and unpredictable
            preprocessor=lambda text: text,
            on_error=lambda: None,
            raise_on_errors=False
    ):
        """
           Initializes a new instance of the ChatGPTBlock class.

           Args:
               system_prompt (str): The system prompt used to guide the conversation.
               examples (Union[None, List[Tuple[Union[str, dict], Union[str, dict]]]], optional): A list of example input-output pairs.
               openai_api_token (Union[str, None], optional): The OpenAI API token. Defaults to None.
               model (str, optional): The GPT model to use. Defaults to 'gpt-4'.
               max_output_length (int, optional): The maximum number of tokens in the generated output. Defaults to 400.
               stream (bool, optional): Whether to use streaming mode. Defaults to True.
               temperature (float, optional): Controls the randomness of the output. Defaults to 0.001.
               preprocessor (callable, optional): A function to preprocess user input. Defaults to the identity text -> text function
               on_error (callable, optional): A function to handle errors. Defaults to an empty function.
               raise_on_errors (bool, optional). Whether raise an exception on OpenAI API error. Defaults to False.
           """
        max_tokens_by_model = {
            'gpt-3.5-turbo': 4097,
            'gpt-3.5-turbo-0301': 4097,
            'gpt-4': 8192,
            'gpt-4-0314': 8192,
        }
        models = list(max_tokens_by_model.keys())
        if model not in models:
            raise KeyError(f'{model} must be in {models}')
        self.stream = stream
        self.model = model
        self.tokens_available = max_tokens_by_model[self.model]
        self.max_output_length = max_output_length
        self.temperature = temperature
        if self.tokens_available < self.max_output_length:
            raise ValueError(
                f'{max_output_length} must be less than max_tokens given by a model. Current: {max_output_length} > {self.tokens_available}')

        openai.api_key = os.getenv('OPENAI_API_KEY') or openai_api_token

        self.system_prompt = {"role": "system", "content": system_prompt}
        self.initial_examples = [
            {"role": role, "content": str(content)}
            for u, a in examples for role, content in [("user", u), ("assistant", a)]
        ] if examples else []
        self.history = []
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:  # TODO: when tiktoken updates its library, remove this try statement
            self.encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')

        _ = self.get_number_of_tokens
        system_prompt_len = _(self.system_prompt["content"])
        examples_len = _(" ".join([x["content"] for x in self.initial_examples]))
        self.tokens_available -= (system_prompt_len + examples_len)
        self.preprocessor = preprocessor
        self._answer = ""
        self.on_error = on_error
        self.raise_on_error = raise_on_errors

    @property
    def history_length(self) -> int:
        """
        Returns the number of tokens in the conversation history.

        Returns:
            int: The number of tokens in the conversation history.
        """
        return self.get_number_of_tokens(self.history)

    @property
    def answer(self) -> str:
        """
        Returns the generated answer.

        Returns:
            str: The generated answer.
        """
        return self._answer

    def get_number_of_tokens(self, input: Union[str, dict, list]) -> int:
        """
        Returns the number of tokens in the input.

        Args:
            input (Union[str, dict, list]): The input for which to count tokens.

        Returns:
            int: The number of tokens in the input.
        """
        if isinstance(input, dict):
            return len(self.encoder.encode(input["content"]))
        elif isinstance(input, str):
            return len(self.encoder.encode(input))
        elif isinstance(input, list):
            return len(self.encoder.encode(' '.join([x['content'] for x in input])))
        else:
            raise NotImplementedError

    def get_trimmed_history(self) -> List:
        """
        Trims the conversation history to fit within the token limit.

        Returns:
            list: The trimmed conversation history.
        """
        total_tokens = self.tokens_available - self.max_output_length
        trimmed_history = []
        tokens_count = 0
        user_element = None

        for element in reversed(self.history):
            tokens_count += self.get_number_of_tokens(element)
            if tokens_count > total_tokens:
                break
            if element["role"] == "user":
                user_element = element
            trimmed_history.insert(0, element)

        if user_element and user_element != trimmed_history[0]:
            index = trimmed_history.index(user_element)
            trimmed_history = trimmed_history[index + 1:]

            new_length = sum(self.get_number_of_tokens(el) for el in trimmed_history)
            logger.info(
                f'history trimmed. New length: {new_length}. Available length: {total_tokens}'
            )

        return trimmed_history

    def call_raw_api(self) -> Union[SimpleStringIterator, GeneratorType, OpenAIObject]:
        """
        Calls the OpenAI API and returns the raw response.

        Returns:
            Union[SimpleStringIterator, GeneratorType, OpenAIObject]: The raw response from the API.
        """

        start = datetime.now()
        try:
            openai_api_response = openai.ChatCompletion.create(
                model=self.model,
                stream=self.stream,
                # max_tokens=self.max_output_length,
                temperature=self.temperature,
                messages=[
                    self.system_prompt,
                    *self.initial_examples,
                    *self.history
                ],
            )
            response = openai_api_response
        except openai.error.OpenAIError as e:
            self.on_error()
            if self.raise_on_error:
                raise e
            errmsg = f"OpenAI internal error. {e}"
            response = SimpleStringIterator(errmsg) if self.stream else errmsg
        except Exception as e:
            self.on_error()
            if self.raise_on_error:
                raise e
            errmsg = f"Internal error. {e}"
            response = SimpleStringIterator(errmsg) if self.stream else errmsg
        end = datetime.now()
        time_elapsed = compute_time_elapsed(start, end)
        logger.debug(f'time waiting for api: {time_elapsed:.3f}')
        return response

    def process_response(self, chunk) -> Union[str, None]:
        """
        Processes a chunk of the API response.

        Args:
            chunk (dict): A chunk of the API response.

        Returns:
            Union[str, None]: The processed response or None.
        """
        chunk = chunk['choices'][0]
        logger.debug(f'chunk: {chunk}')
        finish_reason = chunk["finish_reason"]
        if finish_reason is None:
            content = chunk['delta']['content'] if 'content' in chunk['delta'] else ''
            self._answer += content
            return content
        elif finish_reason == 'stop' or finish_reason == 'length':
            final_answer = None
            if not self.stream and 'message' in chunk and 'content' in chunk['message']:
                self._answer = chunk['message']['content']
                final_answer = self._answer
            self.history.append({"role": "assistant", "content": self._answer})
            self._answer = ""
            return final_answer
        else:
            logger.info(f'finish_reason: {finish_reason}')
            raise Exception(finish_reason)

    def generator_wrapper(self, openai_api_response) -> Generator[str, Any, None]:
        """
        Wraps the generator with error handling.

        Args:
            openai_api_response (GeneratorType): The OpenAI API response generator.

        Yields:
            str: Generated text chunks.
        """
        try:
            for chunk in openai_api_response:
                piece = self.process_response(chunk)
                if piece:
                    yield piece
        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f'(error) {e}')
            self.on_error()
            for chunk in 'openAI error. Please try again.'.split():
                yield chunk + ' '

    def api_answer_wrapper(self) -> Union[SimpleStringIterator, Generator[str, Any, None], str]:
        """
           Wraps the OpenAI API response.

           Returns:
               Union[SimpleStringIterator, Generator[str, Any, None]: The wrapped OpenAI API response.
        """
        openai_api_response = self.call_raw_api()
        if isinstance(openai_api_response, SimpleStringIterator):
            return openai_api_response
        elif isinstance(openai_api_response, GeneratorType):
            return self.generator_wrapper(openai_api_response)
        elif isinstance(openai_api_response, OpenAIObject):
            return self.process_response(openai_api_response)
        else:
            raise ValueError(f"openAI response has an unknown type: {type(openai_api_response)}")

    def __call__(self, *args, **kwargs) -> Union[SimpleStringIterator, GeneratorType, str]:
        """
            Calls the GPT-4 model and returns the generated response.

            Args:
                *args: Variable-length arguments passed to the preprocessor function.
                **kwargs: Keyword arguments passed to the preprocessor function.

            Returns:
                Union[SimpleStringIterator, GeneratorType, str]: The generated response.
        """
        request = self.preprocessor(*args, **kwargs)
        self.history.append({"role": "user", "content": request})
        self.history = self.get_trimmed_history()
        return self.api_answer_wrapper()

    def reset(self):
        """
        Resets the conversation history and generated answer.
        """
        self.history = []
        self._answer = ""
