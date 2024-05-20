# Example Function calling with Gorilla

## Setup

1. Download `gorilla-openfunctions-v2` model from Hugging Face
```shell
huggingface-cli download gorilla-llm/gorilla-openfunctions-v2-gguf gorilla-openfunctions-v2-q4_K_M.gguf
```

2. You may want to symlink to a more convenient location
```shell
ln -s /home/rajan/.cache/huggingface/hub/models--gorilla-llm--gorilla-openfunctions-v2-gguf/snapshots/bab1cf50a4dd54be06c5ea16fd00e60e872bcd9a gorilla-openfunctions-v2-GGUF
```

3. Install llama-cpp-python. Note I am using cuda

```shell
pip install llama-cpp-python   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

4. Run Gorilla Model
```shell
python -m llama_cpp.server --model /home/rajan/models/gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf --chat_format chatml --n_gpu_layers 35 --verbose false
```
This will run llama_cpp.server at port 8000 and OpenAI compatible API available at /v1 end-point


### Multi-model in llama-cpp-python
We can run multiple models in the llama-cpp.server and call the model we are interested in by name.
```shell
python -m llama_cpp.server --config_file config.json
```
