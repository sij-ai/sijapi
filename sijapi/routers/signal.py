"""
Signal Bot example, repeats received messages.
"""
import os

from semaphore import Bot, ChatContext


async def echo(ctx: ChatContext) -> None:
    if not ctx.message.empty():
        await ctx.message.typing_started()
        await ctx.message.reply(ctx.message.get_body())
        await ctx.message.typing_stopped()


async def main() -> None:
    """Start the bot."""
    # Connect the bot to number.
    async with Bot(os.environ["SIGNAL_PHONE_NUMBER"]) as bot:
        bot.register_handler("", echo)

        # Run the bot until you press Ctrl-C.
        await bot.start()


if __name__ == '__main__':
    import anyio
    anyio.run(main)