from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="saiga-mistral-7b.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=0, # Гарантирует использование только CPU
    n_batch=512,
    verbose=True,
    temperature=0.7
)
