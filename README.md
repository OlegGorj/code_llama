# Code LLama POC

## Setup project

```bash
# python -m venv .venv ; source .venv/bin/activate
# source .venv/bin/activate

# poetry env remove python
poetry cache clear . --all
poetry env use python3.11

poetry install -vv
```

Output should be similar to:

```
(main)$ poetry install -vv
The currently activated Python version 3.13.3 is not supported by the project (>=3.11,<3.12).
Trying to find and use a compatible version.
Using python3.11 (3.11.12)
Virtualenv code-llama-2_kZDJSx-py3.11 already exists.
Using virtualenv: /Users/home/Library/Caches/pypoetry/virtualenvs/code-llama-2_kZDJSx-py3.11
Checking keyring availability: Available
Installing dependencies from lock file

Finding the necessary packages for the current system

Package operations: 23 installs, 0 updates, 0 removals

  - Installing markupsafe (3.0.2)
  - Installing mpmath (1.3.0)
  - Installing certifi (2025.4.26)
  - Installing fsspec (2025.3.2)
  - Installing filelock (3.18.0)
  - Installing charset-normalizer (3.4.1)
  - Installing idna (3.10)
  - Installing jinja2 (3.1.6)
  - Installing networkx (3.2.1)
  - Installing sympy (1.13.3)
  - Installing typing-extensions (4.13.2)
  - Installing urllib3 (2.4.0)
  - Installing numpy (1.24.3)
  - Installing six (1.17.0)
  - Installing termcolor (3.0.1)
  - Installing pillow (11.2.1)
  - Installing requests (2.32.3)
  - Installing torch (2.2.0)
  - Installing fairscale (0.4.13)
  - Installing sentencepiece (0.1.99)
  - Installing torchaudio (2.2.0)
  - Installing torchvision (0.17.0)
  - Installing fire (0.5.0)
```

Then, activate the virtual environment:

```bash
source /Users/home/Library/Caches/pypoetry/virtualenvs/code-llama-2_kZDJSx-py3.11/bin/activate
```


### Pretrained Code Models

The Code Llama and Code Llama - Python models are not fine-tuned to follow instructions. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_completion.py` for some examples. To illustrate, see command below to run it with the `CodeLlama-7b` model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_completion.py \
    --ckpt_dir CodeLlama-7b/ \
    --tokenizer_path CodeLlama-7b/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

Pretrained code models are: the Code Llama models `CodeLlama-7b`, `CodeLlama-13b`, `CodeLlama-34b`, `CodeLlama-70b` and the Code Llama - Python models
`CodeLlama-7b-Python`, `CodeLlama-13b-Python`, `CodeLlama-34b-Python`, `CodeLlama-70b-Python`.


### Code Infilling

Code Llama and Code Llama - Instruct 7B and 13B models are capable of filling in code given the surrounding context.


See `example_infilling.py` for some examples. The `CodeLlama-7b` model can be run for infilling with the command below (`nproc_per_node` needs to be set to the `MP` value):
```
torchrun --nproc_per_node 1 example_infilling.py \
    --ckpt_dir CodeLlama-7b/ \
    --tokenizer_path CodeLlama-7b/tokenizer.model \
    --max_seq_len 192 --max_batch_size 4
```

Pretrained infilling models are: the Code Llama models `CodeLlama-7b` and `CodeLlama-13b` and the Code Llama - Instruct models `CodeLlama-7b-Instruct`, `CodeLlama-13b-Instruct`.


### Fine-tuned Instruction Models

Code Llama - Instruct models are fine-tuned to follow instructions. To get the expected features and performance for the 7B, 13B and 34B variants, a specific formatting defined in [`chat_completion()`](https://github.com/facebookresearch/codellama/blob/main/llama/generation.py#L319-L361)
needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and linebreaks in between (we recommend calling `strip()` on inputs to avoid double-spaces).
`CodeLlama-70b-Instruct` requires a separate turn-based prompt format defined in [`dialog_prompt_tokens()`](https://github.com/facebookresearch/codellama/blob/main/llama/generation.py#L506-L548).
You can use `chat_completion()` directly to generate answers with all instruct models; it will automatically perform the required formatting.

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/inference/safety_utils.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using `CodeLlama-7b-Instruct`:

```
torchrun --nproc_per_node 1 example_instructions.py \
    --ckpt_dir CodeLlama-7b-Instruct/ \
    --tokenizer_path CodeLlama-7b-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 4
```

Fine-tuned instruction-following models are: the Code Llama - Instruct models `CodeLlama-7b-Instruct`, `CodeLlama-13b-Instruct`, `CodeLlama-34b-Instruct`, `CodeLlama-70b-Instruct`.

Code Llama is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
In order to help developers address these risks, we have created the [Responsible Use Guide](https://github.com/facebookresearch/llama/blob/main/Responsible-Use-Guide.pdf). More details can be found in our research papers as well.


