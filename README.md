# pdf2gpt-index
build gpt-index using chatgpt and sentence-transformers

## installation 

```bash
    python -m venv env 
    source env/bin/activate
    pip install --upgrade pip
    pip install sentence-transformers click tiktoken pydantic openai protobuf
```

## usage 

```bash
    export OPENAI_API_KEY=sk-
    export TRANSFORMERS_CACHE=/path/to/cache
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

```bash
    python main.py \ 
        build-index --path2pdf_file ressources/code_de_la_famille_senegal.pdf --model_name 'Sahajtomar/french_semantic'\ 
        explore-index --top_k 7
```