import click 
import openai 

import torch as th 

from libraries.strategies import * 

@click.group(chain=True, invoke_without_command=True)
@click.option('--openai_api_key', envvar='OPENAI_API_KEY', required=True)
@click.option('--transformers_cache', envvar='TRANSFORMERS_CACHE', required=True)
@click.pass_context
def command_line_interface(ctx:click.core.Context, openai_api_key:str, transformers_cache:str):
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    ctx.obj['device'] = device
    ctx.obj['tokenizer'] = load_tokenizer(encoding_name='gpt-3.5-turbo')
    ctx.obj['openai_api_key'] = openai_api_key
    ctx.obj['transformers_cache'] = transformers_cache

@command_line_interface.command()
@click.option('--path2pdf_file')
@click.option('--model_name')
@click.pass_context
def build_index(ctx:click.core.Context, path2pdf_file:str, model_name:str):
    tokenizer = ctx.obj['tokenizer']
    transformer = load_transformers(model_name, cache_folder=ctx.obj['transformers_cache'], device=ctx.obj['device'])

    pages = convert_pdf_to_text(path2pdf_file)
    print('nb pages: ', len(pages))
    chunks = split_pages_into_chunks(pages, 256, tokenizer)
    knowledge_base = vectorize(chunks, transformer, device=ctx.obj['device'])

    ctx.obj['knowledge_base'] = knowledge_base  # [(chunk, embedding), ...]
    ctx.obj['transformer'] = transformer


@command_line_interface.command()
@click.option('--top_k', type=int)
@click.pass_context
def explore_index(ctx:click.core.Context, top_k:int):
    openai.api_key = ctx.obj['openai_api_key']
    chunks, embeddings = list(zip(*ctx.obj['knowledge_base']))
    corpus_embeddings = np.vstack(embeddings)
    keep_looping = True 
    while keep_looping:
        try:
            query = input('Enter your query:')
            query_embedding = ctx.obj['transformer'].encode(query, device=ctx.obj['device'])
            paragraphes = find_candidates(
                query_embedding=query_embedding,
                chunks=chunks,
                corpus_embeddings=corpus_embeddings,
                top_k=top_k
            )

            context = []
            for position, chunk in enumerate(paragraphes):
                context.append(f"""
                    Document N°{position}: {chunk}
                """)
            
            context = "###\n\n".join(context)

            print(context)

            completion_response = chatgpt_completion(context, query)
            for chunk in completion_response:
                content = chunk['choices'][0]['delta'].get('content', None)
                if content is not None:
                    print(content, end="", flush=True)
            
            print("\n")

        except KeyboardInterrupt:
            keep_looping = False 
    # end while loop 

if __name__ == '__main__':
    command_line_interface(obj={})