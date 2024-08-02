'''
IN DEVELOPMENT: Retrieval-Augmented Generation module.
NOTES: Haven't yet decided if this should depend on the Obsidian and Chat modules, or if they should depend on it, or one of one the other the other.
'''
#routers/rag.py

from fastapi import APIRouter
from sijapi import L

rag = APIRouter()
logger = L.get_module_logger("rag")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

rag.get("/rag/search")
async def rag_search_endpoint(query: str, scope: str):
    pass

rag.post("/rag/embed")
async def rag_upload_endpoint(path: str):
    pass
