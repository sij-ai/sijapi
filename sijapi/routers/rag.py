'''
IN DEVELOPMENT: Retrieval-Augmented Generation module.
NOTES: Haven't yet decided if this should depend on the Obsidian and Chat modules, or if they should depend on it, or one of one the other the other.
'''

from fastapi import APIRouter

rag = APIRouter()

rag.get("/rag/search")
async def rag_search_endpoint(query: str, scope: str):
    pass

rag.post("/rag/embed")
async def rag_upload_endpoint(path: str):
    pass
