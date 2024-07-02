# cli.py

import click
import asyncio
from datetime import datetime as dt_datetime, timedelta

# Import your async functions and dependencies
from your_main_app import build_daily_note_range_endpoint, loc

def async_command(f):
    @click.command()
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
        async def run():
            return await f(*args, **kwargs)
        return asyncio.run(run())
    return wrapper

@click.group()
def cli():
    """CLI for your application."""
    pass

@cli.command()
@click.argument('dt_start')
@click.argument('dt_end')
@async_command
async def bulk_note_range(dt_start: str, dt_end: str):
    """
    Build daily notes for a date range.
    
    DT_START and DT_END should be in YYYY-MM-DD format.
    """
    try:
        start_date = dt_datetime.strptime(dt_start, "%Y-%m-%d")
        end_date = dt_datetime.strptime(dt_end, "%Y-%m-%d")
    except ValueError:
        click.echo("Error: Dates must be in YYYY-MM-DD format.")
        return

    if start_date > end_date:
        click.echo("Error: Start date must be before or equal to end date.")
        return

    results = []
    current_date = start_date
    while current_date <= end_date:
        formatted_date = await loc.dt(current_date)
        result = await build_daily_note(formatted_date)
        results.append(result)
        current_date += timedelta(days=1)

    click.echo("Generated notes for the following dates:")
    for url in results:
        click.echo(url)

if __name__ == '__main__':
    cli()