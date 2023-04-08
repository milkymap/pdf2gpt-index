import openai 

import PyPDF2

import numpy as np 
import tiktoken


import operator as op 
import itertools as it, functools as ft 

from sentence_transformers import SentenceTransformer

from typing import List, Tuple, Dict, Any, Optional

from model_schema import Message, Role 
from prompt_manager import build_system_settings

def load_tokenizer(encoding_name:str='gpt-3.5-turbo') -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(encoding_name)

def load_transformers(model_name:str, cache_folder:str, device:str='cpu') -> SentenceTransformer:
    return SentenceTransformer(
        model_name_or_path=model_name,
        cache_folder=cache_folder,
        device=device
    )

def convert_pdf_to_text(path2pdf_file:str) -> List[str]:
    reader = PyPDF2.PdfReader(path2pdf_file)
    pages = reader.pages
    accumulator:List[str] = []
    for page in pages:
        text = page.extract_text()
        accumulator.append(text)
    
    return accumulator

def split_pages_into_chunks(pages:List[str], chunk_size:int, tokenizer:tiktoken.Encoding) -> List[str]:
    page_tokens:List[List[int]] = [ tokenizer.encode(page) for page in pages ]
    document_tokens = list(it.chain(*page_tokens))

    nb_tokens = len(document_tokens)
    nb_partitions = round(nb_tokens / chunk_size)

    accumulator:List[str] = []
    for chunk_tokens in np.array_split(document_tokens, nb_partitions):
        paragraph = tokenizer.decode(chunk_tokens)
        accumulator.append(paragraph)
    
    return accumulator

def vectorize(chunks:List[str], transformer:SentenceTransformer, device:str='cpu') -> List[Tuple[str, np.ndarray]]:
    embeddings = transformer.encode(
        sentences=chunks,
        batch_size=32,
        device=device,
        show_progress_bar=True
    )

    return list(zip(chunks, embeddings))

def find_candidates(query_embedding:np.ndarray, chunks:List[str], corpus_embeddings:np.ndarray, top_k:int=7) -> List[str]:
    dot_product = query_embedding @ corpus_embeddings.T 
    norms = np.linalg.norm(query_embedding) * np.linalg.norm(corpus_embeddings, axis=1)
    weighted_scores = dot_product / norms 

    zipped_chunks_scores = list(zip(chunks, weighted_scores))
    sorted_chunks_scores = sorted(zipped_chunks_scores, key=op.itemgetter(1), reverse=True)
    selected_candidates = sorted_chunks_scores[:top_k]

    return list(map(op.itemgetter(0), selected_candidates))

def chatgpt_completion(context:str, query:str):
    completion_rsp = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            build_system_settings(context).dict(),
            Message(
                role=Role.USER,
                content=f"""voici ma question {query}"""
            ).dict()
        ],
        stream=True 
    )

    return completion_rsp







