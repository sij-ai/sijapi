#!/Users/sij/miniforge3/envs/sijapi/bin/python
import sys
import asyncio
from fastapi import BackgroundTasks
from sijapi.routers.news import process_and_save_article

async def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <article_url>")
        sys.exit(1)

    url = sys.argv[1]
    bg_tasks = BackgroundTasks()

    try:
        result = await process_and_save_article(bg_tasks, url)
        print(result)
    except Exception as e:
        print(f"Error processing article: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
