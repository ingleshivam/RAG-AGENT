import ollama

res = ollama.embeddings(
    model='nomic-embed-text:v1.5',
    prompt='hello'
)

print(len(res['embedding']))