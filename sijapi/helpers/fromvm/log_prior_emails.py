import asyncio
from pathlib import Path
from sijapi import L, EMAIL_CONFIG, EMAIL_LOGS
from sijapi.classes import EmailAccount
from sijapi.routers import email

async def initialize_log_files():
    summarized_log = EMAIL_LOGS / "summarized.txt"
    autoresponded_log = EMAIL_LOGS / "autoresponded.txt"
    diagnostic_log = EMAIL_LOGS / "diagnostic.txt"
    for log_file in [summarized_log, autoresponded_log, diagnostic_log]:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("")
    L.DEBUG(f"Log files initialized: {summarized_log}, {autoresponded_log}, {diagnostic_log}")
    return summarized_log, autoresponded_log, diagnostic_log

async def process_all_emails(account: EmailAccount, summarized_log: Path, autoresponded_log: Path, diagnostic_log: Path):
    try:
        with email.get_imap_connection(account) as inbox:
            L.DEBUG(f"Connected to {account.name}, processing all emails...")
            all_messages = inbox.messages()
            unread_messages = set(uid for uid, _ in inbox.messages(unread=True))
            
            processed_count = 0
            for identifier, message in all_messages:
                # Log diagnostic information
                with open(diagnostic_log, 'a') as f:
                    f.write(f"Account: {account.name}, Raw Identifier: {identifier}, Type: {type(identifier)}\n")
                
                # Attempt to get a string representation of the identifier
                if isinstance(identifier, bytes):
                    id_str = identifier.decode()
                elif isinstance(identifier, (int, str)):
                    id_str = str(identifier)
                else:
                    id_str = repr(identifier)
                
                if identifier not in unread_messages:
                    processed_count += 1
                    for log_file in [summarized_log, autoresponded_log]:
                        with open(log_file, 'a') as f:
                            f.write(f"{id_str}\n")
            
            L.INFO(f"Processed {processed_count} non-unread emails for account {account.name}")
    except Exception as e:
        L.logger.error(f"An error occurred while processing emails for account {account.name}: {e}")

async def main():
    email_accounts = email.load_email_accounts(EMAIL_CONFIG)
    summarized_log, autoresponded_log, diagnostic_log = await initialize_log_files()

    L.DEBUG(f"Processing {len(email_accounts)} email accounts")

    tasks = [process_all_emails(account, summarized_log, autoresponded_log, diagnostic_log) for account in email_accounts]
    await asyncio.gather(*tasks)

    # Final verification
    with open(summarized_log, 'r') as f:
        final_count = len(f.readlines())
    L.INFO(f"Final non-unread email count: {final_count}")

if __name__ == "__main__":
    asyncio.run(main())