from faulthandler import disable
from AKB import data, llm, template
import re


def get_query(prompt_gen_template, demos_template, subsampled_data):
    """
    Returns a query for the prompt generator. A query is the prompt that is sent to the LLM.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        subsampled_data: The data to use for the demonstrations.
    Returns:
        A query for the prompt generator.
    """
    # inputs, outputs = subsampled_data
    inputs = [item['entity'] for item in subsampled_data]
    outputs = [item['output'] for item in subsampled_data]
    demos = demos_template.fill((inputs,outputs))
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)


def post_process(prompts, task_type, component):
    for i in range(len(prompts)):
        if component == 'rules':
            try:
                if "[HINTS]" in prompts[i] and "[\\HINTS]" in prompts[i]:
                    prompts[i] = re.search(r"\[HINTS\](.*?)\[\\HINTS\]", prompts[i], re.DOTALL).group(1).strip()
                if "[HINTS]" in prompts[i]:
                    # prompts[i] = re.search(r"\[HINTS\](.*)\n(?!.*\n)", prompts[i], re.DOTALL).group(1).strip()
                    prompts[i] = re.search(r"\[HINTS\](.*)", prompts[i], re.DOTALL).group(1).strip()
            except:
                print(f"[ERROR] in refine_prompt: {prompts[i]}")

            def locate(s, pattern):
                return len(s) if pattern not in s else s.find(pattern)

            if locate(prompts[i], ':') > min(locate(prompts[i], '- '), locate(prompts[i], '1.')) :
                prompts[i] = "Consider the following attributes when making your decision:\n" + prompts[i]

        elif component == 'question':
            try:
                if "[QUESTION]" in prompts[i] and "[\\QUESTION]" in prompts[i]:
                    prompts[i] = re.search(r"\[QUESTION\](.*?)\[\\QUESTION\]", prompts[i], re.DOTALL).group(1).strip()
                if "[QUESTION]" in prompts[i]:
                    # prompts[i] = re.search(r"\[HINTS\](.*)\n(?!.*\n)", prompts[i], re.DOTALL).group(1).strip()
                    prompts[i] = re.search(r"\[QUESTION\](.*)", prompts[i], re.DOTALL).group(1).strip()

                if task_type in [0, 2, 3] and "[Yes, No]" not in prompts[i]: # binary_classification
                    prompts[i] = prompts[i].strip() + "\nChoose your answer from: [Yes, No]"
                if task_type in [0,6] and "[{attribute}: \"{value}\"]" not in prompts[i]: # ED and DC is instrance-wise
                    prompts[i] = "Attribute for Verification: [{attribute}: \"{value}\"]\n" + prompts[i].strip()
                if task_type == 6 and "Answer only" not in prompts[i]:
                    prompts[i] = prompts[i] + "\nAnswer only corrected value of the attribute."
                if task_type == 1 and "only the name of the brand" not in prompts[i]:
                    prompts[i] = prompts[i].strip() + "\nAnswer only the name of the brand."
            except:
                print(f"[ERROR] in refine_prompt: {prompts[i]}")

        elif component == 'task_description':
            try:
                if "[TASK_DESCRIPTION]" in prompts[i] and "[\\TASK_DESCRIPTION]" in prompts[i]:
                    prompts[i] = re.search(r"\[TASK_DESCRIPTION\](.*?)\[\\TASK_DESCRIPTION\]", prompts[i], re.DOTALL).group(1).strip()
                if "[TASK_DESCRIPTION]" in prompts[i]:
                    # prompts[i] = re.search(r"\[HINTS\](.*)\n(?!.*\n)", prompts[i], re.DOTALL).group(1).strip()
                    prompts[i] = re.search(r"\[TASK_DESCRIPTION\](.*)", prompts[i], re.DOTALL).group(1).strip()
            except:
                print(f"[ERROR] in refine_prompt: {prompts[i]}")

    return prompts

def generate_prompts(prompt_gen_template, demos_template, prompt_gen_data, config, component):
    """
    Generates prompts using the prompt generator.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        prompt_gen_data: The data to use for prompt generation.
        config: The configuration dictionary.
    Returns:
        A list of prompts.
    """
    queries = []
    for _ in range(config['num_subsamples']):
        subsampled_data = data.subsample_data(
            prompt_gen_data, config['num_demos'])
        # print(f"[INFO]subsampled_data:{subsampled_data}")
        queries.append(get_query(prompt_gen_template,
                                 demos_template, subsampled_data))

    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    prompts = model.generate_text(
        queries, n=config['num_prompts_per_subsample'])
    del model
    import torch
    torch.cuda.empty_cache()

    prompts = post_process(prompts, prompt_gen_template.task_type, component)
    return prompts


def refine_prompt(prompt_gen_template:template.ErrorTemplate, prompt_gen_data, config):
    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)

    prompts = []
    for _ in range(config['num_subsamples']):
        subsampled_data = data.subsample_data(
                prompt_gen_data, config['num_demos'])
        
        query = prompt_gen_template.prefill(subsampled_data)
        feedback = model.generate_text(
            query, n=1)[0] 

        query = prompt_gen_template.fill(subsampled_data, feedback)
        prompts.extend(model.generate_text(
            query, n=config['num_prompts_per_subsample']))

    del model
    import torch
    torch.cuda.empty_cache()

    # post-progress
    prompts = post_process(prompts, prompt_gen_template.task_type, prompt_gen_template.component)

    return prompts
