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
from datetime import datetime as dt_datetime
from pydantic import BaseModel
from typing import List, Optional, Any
import yaml
from typing import List, Dict, Optional
from pydantic import BaseModel
from sijapi import DEBUG, ERR, LLM_SYS_MSG
from datetime import datetime as dt_datetime
from typing import Dict
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import PODCAST_DIR, DEFAULT_VOICE, EMAIL_CONFIG
from sijapi.routers import tts, llm, sd, locate
from sijapi.utilities import clean_text, assemble_journal_path, extract_text, prefix_lines
from sijapi.classes import EmailAccount, IMAPConfig, SMTPConfig, IncomingEmail, EmailContact


email = APIRouter(tags=["private"])

def load_email_accounts(yaml_path: str) -> List[EmailAccount]:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return [EmailAccount(**account) for account in config['accounts']]


def get_account_by_email(email: str) -> Optional[EmailAccount]:
    email_accounts = load_email_accounts(EMAIL_CONFIG)
    for account in email_accounts:
        if account.imap.username.lower() == email.lower():
            return account
    return None

def get_imap_details(email: str) -> Optional[IMAPConfig]:
    account = get_account_by_email(email)
    return account.imap if account else None

def get_smtp_details(email: str) -> Optional[SMTPConfig]:
    account = get_account_by_email(email)
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

def get_matching_autoresponders(email: IncomingEmail, account: EmailAccount) -> List[Dict]:
    matching_profiles = []

    def matches_list(item: str, email: IncomingEmail) -> bool:
        if '@' in item:
            return item in email.sender
        else:
            return item.lower() in email.subject.lower() or item.lower() in email.body.lower()

    for profile in account.autoresponders:
        whitelist_match = not profile.whitelist or any(matches_list(item, email) for item in profile.whitelist)
        blacklist_match = any(matches_list(item, email) for item in profile.blacklist)

        if whitelist_match and not blacklist_match:
            matching_profiles.append({
                'USER_FULLNAME': account.fullname,
                'RESPONSE_STYLE': profile.style,
                'AUTORESPONSE_CONTEXT': profile.context,
                'IMG_GEN_PROMPT': profile.image_prompt,
                'USER_BIO': account.bio
            })

    return matching_profiles


async def generate_auto_response_body(email: IncomingEmail, profile: Dict) -> str:
    now = await locate.localize_datetime(dt_datetime.now())
    then = await locate.localize_datetime(email.datetime_received)
    age = now - then
    usr_prompt = f'''
Generate a personalized auto-response to the following email:
From: {email.sender}
Sent: {age} ago
Subject: "{email.subject}"
Body:
{email.body}

Respond on behalf of {profile['USER_FULLNAME']}, who is unable to respond personally because {profile['AUTORESPONSE_CONTEXT']}.
Keep the response {profile['RESPONSE_STYLE']} and to the point, but responsive to the sender's inquiry.
Do not mention or recite this context information in your response.
'''
    
    sys_prompt = f"You are an AI assistant helping {profile['USER_FULLNAME']} with email responses. {profile['USER_FULLNAME']} is described as: {profile['USER_BIO']}"
    
    try:
        response = await llm.query_ollama(usr_prompt, sys_prompt, 400)
        DEBUG(f"query_ollama response: {response}")
        
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                ERR(f"Unexpected response structure from query_ollama: {response}")
        else:
            ERR(f"Unexpected response type from query_ollama: {type(response)}")
        
        # If we reach here, we couldn't extract a valid response
        raise ValueError("Could not extract valid response from query_ollama")
        
    except Exception as e:
        ERR(f"Error generating auto-response: {str(e)}")
        return f"Thank you for your email regarding '{email.subject}'. We are currently experiencing technical difficulties with our auto-response system. We will review your email and respond as soon as possible. We apologize for any inconvenience."


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



async def process_account(account: EmailAccount):
    while True:
        start_time = dt_datetime.now()
        try:
            DEBUG(f"Connecting to {account.name} to check for unread emails...")
            with get_imap_connection(account) as inbox:
                DEBUG(f"Connected to {account.name}, checking for unread emails now...")
                unread_messages = inbox.messages(unread=True)
                for uid, message in unread_messages:
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
                    DEBUG(f"\n\nProcessing email for account {account.name}: {this_email.subject}\n\n")
                    save_success = await save_email(this_email, account)
                    respond_success = await autorespond(this_email, account)
                    if save_success and respond_success:
                        inbox.mark_seen(uid)
        except Exception as e:
            ERR(f"An error occurred for account {account.name}: {e}")
        
        # Calculate the time taken for processing
        processing_time = (dt_datetime.now() - start_time).total_seconds()
        
        # Calculate the remaining time to wait
        wait_time = max(0, account.refresh - processing_time)
        
        # Wait for the remaining time
        await asyncio.sleep(wait_time)


async def process_all_accounts():
    email_accounts = load_email_accounts(EMAIL_CONFIG)
    tasks = [asyncio.create_task(process_account(account)) for account in email_accounts]
    await asyncio.gather(*tasks)


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
{email.body}
'''
            
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)

        DEBUG(f"Saved markdown to {md_path}")

        return True
    
    except Exception as e:
        ERR(f"Exception: {e}")
        return False

async def autorespond(this_email: IncomingEmail, account: EmailAccount):
    matching_profiles = get_matching_autoresponders(this_email, account)
    for profile in matching_profiles:
        DEBUG(f"Auto-responding to {this_email.subject} with profile: {profile['USER_FULLNAME']}")
        auto_response_subject = f"Auto-Response Re: {this_email.subject}"
        auto_response_body = await generate_auto_response_body(this_email, profile)
        DEBUG(f"Auto-response: {auto_response_body}")
        await send_auto_response(this_email.sender, auto_response_subject, auto_response_body, profile, account)

async def send_auto_response(to_email, subject, body, profile, account):
    DEBUG(f"Sending auto response to {to_email}...")
    try:
        message = MIMEMultipart()
        message['From'] = account.smtp.username
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        if profile['IMG_GEN_PROMPT']:
            jpg_path = await sd.workflow(profile['IMG_GEN_PROMPT'], earlyout=False, downscale_to_fit=True)
            if jpg_path and os.path.exists(jpg_path):
                with open(jpg_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read(), name=os.path.basename(jpg_path))
                    message.attach(img)

        with get_smtp_connection(account) as server:
            server.login(account.smtp.username, account.smtp.password)
            server.send_message(message)

        INFO(f"Auto-response sent to {to_email} concerning {subject} from account {account.name}")
        return True

    except Exception as e:
        ERR(f"Error in preparing/sending auto-response from account {account.name}: {e}")
        return False

    
@email.on_event("startup")
async def startup_event():
    asyncio.create_task(process_all_accounts())