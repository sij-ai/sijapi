from vectordb import Memory

# Memory is where all content you want to store/search goes.
memory = Memory()

memory.save(
    ["apples are green", "oranges are orange"],  # save your text content. for long text we will automatically chunk it
    [{"url": "https://apples.com"}, {"url": "https://oranges.com"}], # associate any kind of metadata with it (optional)
)

# Search for top n relevant results, automatically using embeddings
query = "green"
results = memory.search(query, top_n = 1)

print(results)