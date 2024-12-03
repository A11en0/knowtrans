import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import PeftModel


def load_model_and_tokenizer(model_weights, lora_weights, enable_vllm=False, tp_size=4, print_param_status=False, use_wlora=True):
    tokenizer = AutoTokenizer.from_pretrained(model_weights)
    # tokenizer.pad_token = "[PAD]"    ### 13B model may has problem here !!!
    # tokenizer.padding_side = "left"
    
    if enable_vllm: 
        model = LLM(model=model_weights, 
                    enable_lora=True if lora_weights else False, 
                    max_lora_rank=64,                    
                    tensor_parallel_size=tp_size, 
                    gpu_memory_utilization=0.7,
                    max_model_len=4096
                )
    else: 
        model = AutoModelForCausalLM.from_pretrained(model_weights).to("cuda")
        
        # model.config.use_cache = False   # for xlora
        print("lora_weights: ", lora_weights)
        if lora_weights: 
            if len(lora_weights) > 1:
                model = PeftModel.from_pretrained(model, lora_weights[0], adapter_name='lora_0')

                for i in range(1, len(lora_weights)):
                    model.load_adapter(lora_weights[i], use_safetensors=True, adapter_name=f"lora_{i}")
                
                adapters = [f'lora_{i}' for i in range(len(lora_weights))]
                weights = [1.0 for _ in range(len(lora_weights))]
                # weights = [1.0, 0, 0, 0]
                model.add_weighted_adapter(
                    adapters=adapters,
                    weights=weights,
                    combination_type="linear",
                    adapter_name="combine"
                )
                model.set_adapter("combine")
            
            if use_wlora:
                lora_weights = lora_weights[0]
                model = PeftModel.from_pretrained(model, os.path.join(lora_weights, 'lora_0'), adapter_name='lora_0') 

                for i in range(1, 11):
                    model.load_adapter(os.path.join(lora_weights, f'lora_{i}'), adapter_name=f'lora_{i}') 
    
    if print_param_status:
        for name, param in model.named_parameters():
            print(name, param.shape)
    
    return model, tokenizer

@torch.inference_mode()
def batch_generate(batch_datas, generation_config, model=None, lora_weights=None, tokenizer=None, enable_vllm=False, is_moe=False, args=None): 
    if enable_vllm:
        sampling_params = SamplingParams(
            temperature=generation_config['temperature'], 
            max_tokens=generation_config['max_tokens'],
            n=generation_config['n'],
            top_p=generation_config['top_p'],
            # use_beam_search=generation_config['use_beam_search'],
            # best_of=generation_config.get('best_of', 1),  # Add best_of parameter
        )
        # print(sampling_params)
        batch_prompts = [item['instruction'] for item in batch_datas]
        
        generated_text = model.generate(
            batch_prompts,
            sampling_params,
            lora_request=LoRARequest("lora", 1, lora_weights[0]) if lora_weights[0] else None # !!! pay attention, there is only one lora weight.
        )
        outputs = []
        for output in generated_text: 
            prompt = output.prompt
            generated_text = output.outputs[0].text.strip()
            outputs.append(generated_text)
            # print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n")
            print(f"Prompt: {prompt}\nGenerated text: {generated_text!r}\n")            
    
    else:
        batch_texts = [item['instruction'] for item in batch_datas]
        print(batch_texts[0])
        batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=generation_config['max_tokens']).to(model.device)
        
        if is_moe:
            batch_inputs['task_ids'] = torch.LongTensor([data['task_type'] for data in batch_datas]).to(model.device)
        
        if args.model_type == 'mistral': 
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("[PAD]")
            ]

        if args.model_type == 'llama3': 
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ] 
        
        if args.model_type == 'jellyfish-13b': 
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|end|>")
            ] 
        
        generation_outputs = model.generate(
            **batch_inputs,
            generation_config=GenerationConfig(**generation_config),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            num_return_sequences=generation_config.get('best_of', 1)  # Add best_of parameter
        )
        
        if 'CausalLM' in str(model.__class__):
            outputs = tokenizer.batch_decode(generation_outputs[:, batch_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        else:
            outputs = tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)

        # Select the best output
        if generation_config.get('best_of', 1) > 1:
            outputs = [max(outputs[i:i+generation_config['best_of']], key=lambda x: model(**tokenizer(x, return_tensors="pt").to(model.device)).logits.mean().item()) for i in range(0, len(outputs), generation_config['best_of'])]

    return outputs