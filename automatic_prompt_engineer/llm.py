"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

import openai
from openai import OpenAI
from vllm import LLM as vLLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import gc
import random
import re

gpt_costs_per_thousand = {
    'davinci': 0.0200,
    'curie': 0.0020,
    'babbage': 0.0005,
    'ada': 0.0004
}

# 全局字典用于存储已经创建的模型对象
_model_cache = {}

def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    model_name = config['gpt_config']['model']
    if model_name in _model_cache:
        return _model_cache[model_name]
    else:
        for name in _model_cache.keys():
            _model_cache[name].cleanup()
        _model_cache.clear()
        torch.cuda.empty_cache()
    if model_type == "GPT_forward":
        model = GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        raise ValueError(f"[ERROR] insert mode is deperacted.")
        model = GPT_Insert(config, disable_tqdm=disable_tqdm)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    _model_cache[model_name] = model
    return model


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        """Returns the log probs of the text.
        Parameters:
            text: The text to get the log probs of. This can be a string or a list of strings.
            log_prob_range: The range of characters within each string to get the log_probs of. 
                This is a list of tuples of the form (start, end).
        Returns:
            A list of log probs.
        """
        pass


class GPT_Forward(LLM):
    """Wrapper for GPT-3."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        self.model_id = config['gpt_config']['model']
        if 'gpt' in self.model_id:
            self.client = OpenAI(
                # base_url = 'http://a40c09:8000/v1',
                # api_key=''
            )
            # temp added for token analysis
            self.tokenizer = AutoTokenizer.from_pretrained('/share/home/12351018/pre-train/Jellyfish-7B')
        else:
            
            if '7B' in self.model_id:
                self.tokenizer = AutoTokenizer.from_pretrained('/share/home/12351018/pre-train/Jellyfish-7B')
            elif '8B' in self.model_id:
                self.tokenizer = AutoTokenizer.from_pretrained('/share/home/12351018/pre-train/Jellyfish-8B')
            elif '13B' in self.model_id:
                self.tokenizer = AutoTokenizer.from_pretrained('/share/home/12351018/pre-train/Jellyfish-13B')
            elif 'TableLlama' in self.model_id:
                self.tokenizer = AutoTokenizer.from_pretrained('/share/home/12351018/pre-train/TableLlama')
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

            self.lora_path = config['gpt_config'].get('lora', None)
            if self.lora_path and ('--merge' in self.lora_path or '-weight' in self.lora_path): # merged lora cannot use vllm directly
                if '13B' in self.model_id:
                    self.model = AutoModelForCausalLM.from_pretrained(config['gpt_config']['model'], torch_dtype=torch.float16).to("cuda")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(config['gpt_config']['model']).to("cuda")#, torch_dtype=torch.float16
                self.model = PeftModel.from_pretrained(self.model, os.path.join(self.lora_path, 'lora_0'), adapter_name='lora_0') 
                for i in range(1, 11):
                    self.model.load_adapter(os.path.join(self.lora_path, f'lora_{i}'), adapter_name=f'lora_{i}') 
            else:
                self.vllm = vLLM(model=config['gpt_config']['model'], 
                    enable_lora=True if self.lora_path else False, 
                    max_lora_rank=64,                    
                    tensor_parallel_size=1, 
                    gpu_memory_utilization=0.9,
                    max_model_len=4096
                )
        
    def __del__(self):
        """Destructor to free resources."""
        if hasattr(self, 'vllm'):
            del self.vllm  # 删除vLLM实例
            del self.tokenizer
            torch.cuda.empty_cache()  # 清空CUDA缓存

    def cleanup(self):
        if hasattr(self, 'vllm'):
            del self.vllm
            del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '')#.strip()
        responses = []
        sampling_params = SamplingParams(
            temperature=config.get('temperature', 0), 
            top_p=config.get('top_p', 1), 
            top_k=config.get('top_k', 50),
            max_tokens=config.get('max_tokens', 20), 
            use_beam_search=config.get('use_beam_search', False), 
            best_of=config.get('best_of', 1), 
            seed=config.get('seed', 42), 
            stop=["[INST]","\n\nRecord [","<|eot_id|>", "\n\n\n"],
        )
        generation_config = {
            "n": 1, 
            "temperature": sampling_params.temperature,
            "max_tokens": 4000,
            "top_p": sampling_params.top_p, 
            "top_k": sampling_params.top_k, 
            "max_new_tokens": sampling_params.max_tokens, 
            "do_sample": True,
            "use_beam_search": False,
            "best_of": 1  # Add best_of parameter
            # "repetition_penalty": 1.3   # harmful to performance !!!
        }
        system_message = "You are an AI assistant that follows instruction extremely well. User will give you a question. Your task is to answer as faithfully as you can." 
        if "TableLlama" in self.model_id:
            system_message = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        while len(responses) < len(prompt) * n:
            try:
                # response = openai.Completion.create(
                #     **config, prompt=prompt)
                if hasattr(self, 'client'):
                    if len(prompt) > 1:
                        raise ValueError(f"[ERROR] openai api does not support batch inferrring.")
                    token_ids = self.tokenizer(prompt[0], add_special_tokens=False)["input_ids"]
                    print(f"[INPUT TOKEN] {len(token_ids)}")
                    response = self.client.chat.completions.create(
                        **config,
                        messages=[{"role": "user", "content": prompt[0]}]
                    )
                    responses.extend([res.message.content for res in response.choices])
                    token_ids = self.tokenizer(responses[-1], add_special_tokens=False)["input_ids"]
                    print(f"[OUTPUT TOKEN] {len(token_ids)}")
                
                elif hasattr(self, 'model'):
                    if '7B' in self.model_id:
                        message = [f"{system_message}\n\n[INST]\n\n{pt}\n\n[/INST]\n\n" for pt in prompt]
                        batch_inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=generation_config['max_tokens']).to("cuda")
                        terminators = [
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("[PAD]")
                        ]
                        outputs = self.model.generate(
                            **batch_inputs,
                            generation_config=GenerationConfig(**generation_config),
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=terminators,
                            num_return_sequences=generation_config['best_of']  # Add best_of parameter
                        )
                    elif '13B' in self.model_id:
                        message = [f"{system_message}\n\n### Instruction:\n\n{pt}\n\n### Response:\n\n" for pt in prompt]
                        batch_inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=generation_config['max_tokens']).to("cuda")
                        outputs = self.model.generate(
                            **batch_inputs,
                            generation_config=GenerationConfig(**generation_config),
                            pad_token_id=self.tokenizer.eos_token_id,
                            num_return_sequences=generation_config['best_of']  # Add best_of parameter
                        )
                    elif '8B' in self.model_id:
                        message = [
                            [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content":pt},
                            ] 
                            for pt in prompt
                        ]
                        # self.tokenizer.pad_token_id = '[PAD]'
                        message = self.tokenizer.apply_chat_template(
                            message,
                            add_generation_prompt=True,
                            tokenize=False,
                            return_tensors="pt",
                            # padding=True,
                            # max_length=generation_config['max_tokens']
                        )#.to("cuda")
                        batch_inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=generation_config['max_tokens']).to("cuda")

                        terminators = [
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
                        outputs = self.model.generate(
                            **batch_inputs,
                            generation_config=GenerationConfig(**generation_config),
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                            num_return_sequences=generation_config['best_of']  # Add best_of parameter
                        )
                    responses = self.tokenizer.batch_decode(outputs[:, batch_inputs['input_ids'].shape[1]:], skip_specical_tokens=True)
                    responses = [re.sub(r"<\|eot_id\|>+", "", response) for response in responses]# 使用正则表达式去除末尾连续的 `<|eot_id|>` 标签
                    responses = [re.sub(r"</s>+", "", response) for response in responses]# 使用正则表达式去除末尾连续的 `</s>` 标签
                else:
                    if '8B' in self.model_id:
                        message = [
                            [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content":pt},
                            ] 
                            for pt in prompt
                        ]
                        input_ids = self.tokenizer.apply_chat_template(
                            message,
                            add_generation_prompt=True,
                            tokenize=False,
                            # return_tensors="pt",
                            # padding=True,
                        )
                        # response = self.vllm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
                        response = self.vllm.generate(prompts=input_ids, 
                                                      sampling_params=sampling_params,
                                                      lora_request=LoRARequest("default", 1, self.lora_path) if self.lora_path else None,
                                                      )
                    # else:
                    #     message = [
                    #         [
                    #             {"role": "user", "content": f"{system_message}\n\n{pt}"}, # [INST] should behind the system_message
                    #         ] 
                    #         for pt in prompt
                    #     ]
                    #     input_ids = self.tokenizer.apply_chat_template(
                    #         message,
                    #         add_generation_prompt=True,
                    #         # return_tensors="pt",
                    #         # padding=True,
                    #     )
                    #     response = self.vllm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
                    else:
                        if '13B' in self.model_id:
                            message = [f"{system_message}\n\n### Instruction:\n\n{pt}\n\n### Response:\n\n" for pt in prompt]
                        elif 'TableLlama' in self.model_id:
                            message = [f"{system_message}\n\n### Instruction:\n{pt}\n\n### Response:\n" for pt in prompt]
                        else:
                            message = [f"{system_message}\n\n[INST]\n\n{pt}\n\n[/INST]\n\n" for pt in prompt]
                        response = self.vllm.generate(prompts=message, sampling_params=sampling_params, lora_request=LoRARequest("default", 1, self.lora_path) if self.lora_path else None)
                    responses.extend([res.outputs[0].text for res in responses])
                if random.randint(0, 9) == 0:
                    print(f"[INFO] prompt in GPT_Forward.__generate_text: {message}")
                    print(f"[INFO] last response is :{responses[-1]}")
            except Exception as e:
                if 'is greater than the maximum' in str(e):
                    raise BatchSizeException()
                print(e)
                print('Retrying...')
                time.sleep(10)
    
        # return [response.choices[i]['text'] for i in range(len(response['choices']))]
        # return [response.choices[0].message.content]
        return responses

    def __complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        response = None
        while response is None:
            try:
                raise ValueError(f"[ERROR] the complete mode is deprecated")
                response = openai.Completion.create(
                    **config, prompt=prompt)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        return response['choices']
        # return completion.choices[0].message

    def __log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if log_prob_range is not None:
            for i in range(len(text)):
                lower_index, upper_index = log_prob_range[i]
                assert lower_index < upper_index
                assert lower_index >= 0
                assert upper_index - 1 < len(text[i])
        config = self.config['gpt_config'].copy()
        # config['logprobs'] = 1
        # config['echo'] = True # deprecated
        # config['max_tokens'] = 1 #[change] 'max_tokens must be at least 1, got 0.'
        # if isinstance(text, list):
        #     text = [f'\n{text[i]}' for i in range(len(text))]
        # else:
        #     text = f'\n{text}'
        response = None
        sampling_params = SamplingParams(
            temperature=1, 
            top_p=config.get('top_p', 1), 
            top_k=5,
            max_tokens=3, 
            use_beam_search=config.get('use_beam_search', False), 
            best_of=config.get('best_of', 1), 
            seed=config.get('seed', 42), 
            stop=["[INST]","\n\nRecord","<|eot_id|>"],
            prompt_logprobs=5, # max 5
            # logprobs=5,
        )
        tokenizer = self.vllm.get_tokenizer()
        query_token_ids = tokenizer(text)['input_ids']
        inputs = [txt[:rg[0]] for txt,rg in zip(text,log_prob_range)]
        input_token_ids = tokenizer(inputs)['input_ids']

        while response is None:
            # response = openai.Completion.create(
            #     **config, prompt=text)
            response = self.vllm.generate(prompt_token_ids=query_token_ids, sampling_params=sampling_params)

            if random.randint(0, 9) == 0:
                print(f"[INFO] prompt in GPT_Forward.__log_probs: {text}")
                print(f"[INFO] last response is :{response[-1]}")

        # log_probs = [response['choices'][i]['logprobs']['token_logprobs'][1:]
        #              for i in range(len(response['choices']))]
        # tokens = [response['choices'][i]['logprobs']['tokens'][1:]
        #           for i in range(len(response['choices']))]
        # offsets = [response['choices'][i]['logprobs']['text_offset'][1:]
        #            for i in range(len(response['choices']))]
        
        log_probs = []
        tokens = []

        for i in range(len(response)):
            input_length = len(input_token_ids[i])
            output_token_ids = query_token_ids[i][input_length:]
            output_logprobs = response[i].prompt_logprobs[input_length:]
            # output_logprobs = response[i].outputs
            log_probs_batch = []
            tokens_batch = []
            offsets_batch = []

            for idx, token_id in enumerate(output_token_ids):
                log_probs_batch.append(getattr(output_logprobs[idx][token_id], 'logprob') if token_id in output_logprobs[idx] else float('-inf'))
                tokens_batch.append(token_id)

            log_probs.append(log_probs_batch)
            tokens.append(tokens_batch)
                

        # # Subtract 1 from the offsets to account for the newline
        # for i in range(len(offsets)):
        #     offsets[i] = [offset - 1 for offset in offsets[i]]

        # if log_prob_range is not None:
        #     # First, we need to find the indices of the tokens in the log probs
        #     # that correspond to the tokens in the log_prob_range
        #     for i in range(len(log_probs)):
        #         lower_index, upper_index = self.get_token_indices(
        #             offsets[i], log_prob_range[i])
        #         log_probs[i] = log_probs[i][lower_index:upper_index]
        #         tokens[i] = tokens[i][lower_index:upper_index]

        del tokenizer
        return log_probs, tokens

    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range."""
        # For the lower index, find the highest index that is less than or equal to the lower index
        lower_index = 0
        for i in range(len(offsets)):
            if offsets[i] <= log_prob_range[0]:
                lower_index = i
            else:
                break

        upper_index = len(offsets)
        for i in range(len(offsets)):
            if offsets[i] >= log_prob_range[1]:
                upper_index = i
                break

        return lower_index, upper_index


class GPT_Insert(LLM):
    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        assert batch_size == 1
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, split into {len(prompt_batches)} batches of (maximum) size {batch_size * n}")
        text = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def log_probs(self, text, log_prob_range=None):
        raise NotImplementedError

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # Split prompts into prefixes and suffixes with the [APE] token (do not include the [APE] token in the suffix)
        prefix = prompt[0].split('[APE]')[0]
        suffix = prompt[0].split('[APE]')[1]
        print(f"[DEBUG] prefix: {prefix}")
        print(f"[DEBUG] suffix: {suffix}")
        response = None
        while response is None:
            try:
                openai.api_key = ''
                response = openai.Completion.create(
                    **config, prompt=prefix, suffix=suffix)
                # from openai import OpenAI
                # client = OpenAI()
                # completion = client.chat.completions.create(
                #     **config,
                #     model="text-davinci-002",
                #     messages=[
                #         # {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                #         {"role": "user", "content": prefix}
                #     ],
                #     suffix=suffix,
                # )
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)

        # Remove suffix from the generated text
        texts = [response.choices[i]['text'].replace(suffix, '') for i in range(len(response['choices']))]
        return texts


def gpt_get_estimated_cost(config, prompt, max_tokens):
    """Uses the current API costs/1000 tokens to estimate the cost of generating text from the model."""
    # Get rid of [APE] token
    prompt = prompt.replace('[APE]', '')
    # Get the number of tokens in the prompt
    n_prompt_tokens = len(prompt) // 4
    # Get the number of tokens in the generated text
    total_tokens = n_prompt_tokens + max_tokens
    engine = config['gpt_config']['model'].split('-')[1]
    costs_per_thousand = gpt_costs_per_thousand
    if engine not in costs_per_thousand:
        # Try as if it is a fine-tuned model
        engine = config['gpt_config']['model'].split(':')[0]
        costs_per_thousand = {
            'davinci': 0.1200,
            'curie': 0.0120,
            'babbage': 0.0024,
            'ada': 0.0016
        }
    price = costs_per_thousand[engine] * total_tokens / 1000
    return price


class BatchSizeException(Exception):
    pass
