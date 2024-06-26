'''
Uses IMAP and SMTP login credentials to monitor an inbox and summarize incoming emails that match certain criteria and save the Text-To-Speech converted summaries into a specified "podcast" folder. 
'''
from fastapi import APIRouter
import asyncio
from imbox import Imbox
from bs4 import BeautifulSoup
import os
from pathlib import Path
from shutil import move
import tempfile
import re
from smtplib import SMTP_SSL, SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import ssl
import yaml
from typing import List, Dict, Optional, Set
from datetime import datetime as dt_datetime
from sijapi import L, PODCAST_DIR, DEFAULT_VOICE, EMAIL_CONFIG, EMAIL_LOGS
from sijapi.routers import tts, llm, sd, locate
from sijapi.utilities import clean_text, assemble_journal_path, extract_text, prefix_lines
from sijapi.classes import EmailAccount, IMAPConfig, SMTPConfig, IncomingEmail, EmailContact, AutoResponder
from sijapi.classes import EmailAccount

email = APIRouter(tags=["private"])

def load_email_accounts(yaml_path: str) -> List[EmailAccount]:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return [EmailAccount(**account) for account in config['accounts']]


def get_account_by_email(this_email: str) -> Optional[EmailAccount]:
    email_accounts = load_email_accounts(EMAIL_CONFIG)
    for account in email_accounts:
        if account.imap.username.lower() == this_email.lower():
            return account
    return None

def get_imap_details(this_email: str) -> Optional[IMAPConfig]:
    account = get_account_by_email(this_email)
    return account.imap if account else None

def get_smtp_details(this_email: str) -> Optional[SMTPConfig]:
    account = get_account_by_email(this_email)
    return account.smtp if account else None


def get_imap_connection(account: EmailAccount):
    return Imbox(account.imap.host,
        username=account.imap.username,
        password=account.imap.password,
        port=account.imap.port,
        ssl=account.imap.encryption == 'SSL',
        starttls=account.imap.encryption == 'STARTTLS')

def get_smtp_connection(account: EmailAccount):
    context = ssl._create_unverified_context()
    
    if account.smtp.encryption == 'SSL':
        return SMTP_SSL(account.smtp.host, account.smtp.port, context=context)
    elif account.smtp.encryption == 'STARTTLS':
        smtp = SMTP(account.smtp.host, account.smtp.port)
        smtp.starttls(context=context)
        return smtp
    else:
        return SMTP(account.smtp.host, account.smtp.port)


def get_matching_autoresponders(this_email: IncomingEmail, account: EmailAccount) -> List[AutoResponder]:
    L.DEBUG(f"Called get_matching_autoresponders for email \"{this_email.subject},\" account name \"{account.name}\"")
    def matches_list(item: str, this_email: IncomingEmail) -> bool:
        if '@' in item:
            return item in this_email.sender
        else:
            return item.lower() in this_email.subject.lower() or item.lower() in this_email.body.lower()
    matching_profiles = []
    for profile in account.autoresponders:
        whitelist_match = not profile.whitelist or any(matches_list(item, this_email) for item in profile.whitelist)
        blacklist_match = any(matches_list(item, this_email) for item in profile.blacklist)
        if whitelist_match and not blacklist_match:
            L.DEBUG(f"We have a match for {whitelist_match} and no blacklist matches.")
            matching_profiles.append(profile)
        elif whitelist_match and blacklist_match:
            L.DEBUG(f"Matched whitelist for {whitelist_match}, but also matched blacklist for {blacklist_match}")
        else:
            L.DEBUG(f"No whitelist or blacklist matches.")
    return matching_profiles


async def generate_auto_response_body(this_email: IncomingEmail, profile: AutoResponder, account: EmailAccount) -> str:
    now = await locate.localize_datetime(dt_datetime.now())
    then = await locate.localize_datetime(this_email.datetime_received)
    age = now - then
    usr_prompt = f'''
    Generate a personalized auto-response to the following email:
    From: {this_email.sender}
    Sent: {age} ago
    Subject: "{this_email.subject}"
    Body:
    {this_email.body}
    Respond on behalf of {account.fullname}, who is unable to respond personally because {profile.context}.
    Keep the response {profile.style} and to the point, but responsive to the sender's inquiry.
    Do not mention or recite this context information in your response.
    '''
    sys_prompt = f"You are an AI assistant helping {account.fullname} with email responses. {account.fullname} is described as: {account.bio}"
    try:
        # async def query_ollama(usr: str, sys: str = LLM_SYS_MSG, model: str = DEFAULT_LLM, max_tokens: int = 200):
        response = await llm.query_ollama(usr_prompt, sys_prompt, profile.ollama_model, 400)

        L.DEBUG(f"query_ollama response: {response}")
        
        if isinstance(response, str):
            response += "\n\n"
            return response
        elif isinstance(response, dict):
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                L.ERR(f"Unexpected response structure from query_ollama: {response}")
        else:
            L.ERR(f"Unexpected response type from query_ollama: {type(response)}")
        
        # If we reach here, we couldn't extract a valid response
        raise ValueError("Could not extract valid response from query_ollama")
        
    except Exception as e:
        L.ERR(f"Error generating auto-response: {str(e)}")
        return f"Thank you for your email regarding '{this_email.subject}'. We are currently experiencing technical difficulties with our auto-response system. We will review your email and respond as soon as possible. We apologize for any inconvenience."


def clean_email_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return re.sub(r'[ \t\r\n]+', ' ', soup.get_text()).strip()


async def extract_attachments(attachments) -> List[str]:
    attachment_texts = []
    for attachment in attachments:
        attachment_name = attachment.get('filename', 'tempfile.txt')
        _, ext = os.path.splitext(attachment_name)
        ext = ext.lower() if ext else '.txt'
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_file.write(attachment['content'].getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            attachment_text = await extract_text(tmp_file_path)
            attachment_texts.append(attachment_text)
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    return attachment_texts


async def save_email(this_email: IncomingEmail, account: EmailAccount):
    try:
        md_path, md_relative = assemble_journal_path(this_email.datetime_received, "Emails", this_email.subject, ".md")
        tts_path, tts_relative = assemble_journal_path(this_email.datetime_received, "Emails", this_email.subject, ".wav")
        summary = ""
        if account.summarize == True:
            email_content = f'At {this_email.datetime_received}, {this_email.sender} sent an email with the subject line "{this_email.subject}". The email in its entirety reads: \n\n{this_email.body}\n"'
            if this_email.attachments:
                attachment_texts = await extract_attachments(this_email.attachments)
                email_content += "\n—--\n" + "\n—--\n".join([f"Attachment: {text}" for text in attachment_texts])
            summary = await llm.summarize_text(email_content)
            await tts.local_tts(text_content = summary, speed = 1.1, voice = DEFAULT_VOICE, podcast = account.podcast, output_path = tts_path)
            summary = prefix_lines(summary, '> ')
        
        # Create the markdown content
        markdown_content = f'''---
date: {this_email.datetime_received.strftime('%Y-%m-%d')}
tags:
- email
---
|     |     |     | 
| --: | :--: |  :--: | 
|  *received* | **{this_email.datetime_received.strftime('%B %d, %Y at %H:%M:%S %Z')}**    |    |
|  *from* | **[[{this_email.sender}]]**    |    |
|  *to* | {', '.join([f'**[[{recipient.email}]]**' if not recipient.name else f'**[[{recipient.name}|{recipient.email}]]**' for recipient in this_email.recipients])}   |    |
|  *subject* | **{this_email.subject}**    |    |
'''
    
        if summary:
            markdown_content += f'''
> [!summary]  Summary
>  {summary}
'''
  
        if tts_path.exists():
            markdown_content += f'''
![[{tts_path}]]
'''
            
        markdown_content += f'''
---
{this_email.body}
'''
            
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)

        L.INFO(f"Saved markdown to {md_path}")

        return True
    
    except Exception as e:
        L.ERR(f"Exception: {e}")
        return False

async def autorespond(this_email: IncomingEmail, account: EmailAccount):
    L.DEBUG(f"Evaluating {this_email.subject} for autoresponse-worthiness...")
    matching_profiles = get_matching_autoresponders(this_email, account)
    L.DEBUG(f"Matching profiles: {matching_profiles}")
    for profile in matching_profiles:
        L.INFO(f"Generating auto-response to {this_email.subject} with profile: {profile.name}")
        auto_response_subject = f"Auto-Response Re: {this_email.subject}"
        auto_response_body = await generate_auto_response_body(this_email, profile, account)
        L.DEBUG(f"Auto-response: {auto_response_body}")
        await send_auto_response(this_email.sender, auto_response_subject, auto_response_body, profile, account)

async def send_auto_response(to_email, subject, body, profile, account):
    try:
        message = MIMEMultipart()
        message['From'] = account.smtp.username
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        if profile.image_prompt:
            jpg_path = await sd.workflow(profile.image_prompt, earlyout=False, downscale_to_fit=True)
            if jpg_path and os.path.exists(jpg_path):
                with open(jpg_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read(), name=os.path.basename(jpg_path))
                    message.attach(img)

        L.DEBUG(f"Sending auto-response {to_email} concerning {subject} from account {account.name}...")
        with get_smtp_connection(account) as server:
            server.login(account.smtp.username, account.smtp.password)
            server.send_message(message)

        L.INFO(f"Auto-response sent to {to_email} concerning {subject} from account {account.name}!")
        return True

    except Exception as e:
        L.ERR(f"Error in preparing/sending auto-response from account {account.name}: {e}")
        return False




async def load_processed_uids(filename: Path) -> Set[str]:
    if filename.exists():
        with open(filename, 'r') as f:
            return set(line.strip().split(':')[-1] for line in f)
    return set()

async def save_processed_uid(filename: Path, account_name: str, uid: str):
    with open(filename, 'a') as f:
        f.write(f"{account_name}:{uid}\n")

async def process_account_summarization(account: EmailAccount):
    summarized_log = EMAIL_LOGS / "summarized.txt"

    while True:
        try:
            processed_uids = await load_processed_uids(summarized_log)
            with get_imap_connection(account) as inbox:
                unread_messages = inbox.messages(unread=True)
                for uid, message in unread_messages:
                    uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                    if uid_str not in processed_uids:
                        recipients = [EmailContact(email=recipient['email'], name=recipient.get('name', '')) for recipient in message.sent_to]
                        localized_datetime = await locate.localize_datetime(message.date)
                        this_email = IncomingEmail(
                            sender=message.sent_from[0]['email'],
                            datetime_received=localized_datetime,
                            recipients=recipients,
                            subject=message.subject,
                            body=clean_email_content(message.body['html'][0]) if message.body['html'] else clean_email_content(message.body['plain'][0]) or "",
                            attachments=message.attachments
                        )
                        if account.summarize:
                            save_success = await save_email(this_email, account)
                            if save_success:
                                await save_processed_uid(summarized_log, account.name, uid_str)
                                L.INFO(f"Summarized email: {uid_str}")
        except Exception as e:
            L.ERR(f"An error occurred during summarization for account {account.name}: {e}")
        
        await asyncio.sleep(account.refresh)

async def process_account_autoresponding(account: EmailAccount):
    autoresponded_log = EMAIL_LOGS / "autoresponded.txt"

    while True:
        try:
            processed_uids = await load_processed_uids(autoresponded_log)
            L.DEBUG(f"{len(processed_uids)} already processed emails are being ignored.")
            with get_imap_connection(account) as inbox:
                unread_messages = inbox.messages(unread=True)
                for uid, message in unread_messages:
                    uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                    if uid_str not in processed_uids:
                        recipients = [EmailContact(email=recipient['email'], name=recipient.get('name', '')) for recipient in message.sent_to]
                        localized_datetime = await locate.localize_datetime(message.date)
                        this_email = IncomingEmail(
                            sender=message.sent_from[0]['email'],
                            datetime_received=localized_datetime,
                            recipients=recipients,
                            subject=message.subject,
                            body=clean_email_content(message.body['html'][0]) if message.body['html'] else clean_email_content(message.body['plain'][0]) or "",
                            attachments=message.attachments
                        )
                        L.DEBUG(f"Attempting autoresponse on {this_email.subject}")
                        respond_success = await autorespond(this_email, account)
                        if respond_success:
                            await save_processed_uid(autoresponded_log, account.name, uid_str)
                            L.WARN(f"Auto-responded to email: {uid_str}")
        except Exception as e:
            L.ERR(f"An error occurred during auto-responding for account {account.name}: {e}")
        
        await asyncio.sleep(account.refresh)

async def process_all_accounts():
    
    email_accounts = load_email_accounts(EMAIL_CONFIG)
    summarization_tasks = [asyncio.create_task(process_account_summarization(account)) for account in email_accounts]
    autoresponding_tasks = [asyncio.create_task(process_account_autoresponding(account)) for account in email_accounts]
    await asyncio.gather(*summarization_tasks, *autoresponding_tasks)

@email.on_event("startup")
async def startup_event():
    await asyncio.sleep(5)
    asyncio.create_task(process_all_accounts())