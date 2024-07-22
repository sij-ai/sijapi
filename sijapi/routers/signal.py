"""
Signal Bot example, repeats received messages.
"""
import os
from fastapi import APIRouter
from semaphore import Bot, ChatContext
from sijapi import L

signal = APIRouter()

logger = L.get_module_logger("signal")

def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

async def echo(ctx: ChatContext) -> None:
    if not ctx.message.empty():
        await ctx.message.typing_started()
        await ctx.message.reply(ctx.message.get_body())
        await ctx.message.typing_stopped()

async def main() -> None:
    """Start the bot."""
    async with Bot(os.environ["SIGNAL_PHONE_NUMBER"]) as bot:
        bot.register_handler("", echo)
        await bot.start()

if __name__ == '__main__':
    import anyio
    anyio.run(main)