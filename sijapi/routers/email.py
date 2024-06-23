'''
IN DEVELOPMENT Email module. Uses IMAP and SMTP login credentials to monitor an inbox and summarize incoming emails that match certain criteria and save the Text-To-Speech converted summaries into a specified "podcast" folder. 
UNIMPLEMENTED: AI auto-responder.
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
import ssl
from smtplib import SMTP_SSL
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime as dt_datetime
from pydantic import BaseModel
from typing import List, Optional, Any
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import HOME_DIR, DATA_DIR, OBSIDIAN_VAULT_DIR, PODCAST_DIR, IMAP, OBSIDIAN_JOURNAL_DIR, DEFAULT_VOICE, AUTORESPONSE_BLACKLIST, AUTORESPONSE_WHITELIST, AUTORESPONSE_CONTEXT, USER_FULLNAME, USER_BIO, AUTORESPOND, TZ
from sijapi.routers import summarize, tts, llm
from sijapi.utilities import clean_text, assemble_journal_path, localize_dt, extract_text, prefix_lines


email = APIRouter(tags=["private"])


class Contact(BaseModel):
    email: str
    name: str
class EmailModel(BaseModel):
    sender: str
    recipients: List[Contact]
    datetime_received: dt_datetime
    subject: str
    body: str
    attachments: Optional[List[Any]] = None

def imap_conn():
    return Imbox(IMAP.host,
        username=IMAP.email,
        password=IMAP.password,
        port=IMAP.imap_port,
        ssl=IMAP.imap_encryption == 'SSL',
        starttls=IMAP.imap_encryption == 'STARTTLS')


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


async def process_unread_emails(auto_respond: bool = AUTORESPOND, summarize_emails: bool = True, podcast: bool = True):
    while True:
        try:
            with imap_conn() as inbox:
                unread_messages = inbox.messages(unread=True)
                for uid, message in unread_messages:
                    recipients = [Contact(email=recipient['email'], name=recipient.get('name', '')) for recipient in message.sent_to]
                    this_email = EmailModel(
                        sender=message.sent_from[0]['email'],
                        datetime_received=localize_dt(message.date),
                        recipients=recipients,
                        subject=message.subject,
                        body=clean_email_content(message.body['html'][0]) if message.body['html'] else clean_email_content(message.body['plain'][0]) or "",
                        attachments=message.attachments
                    )

                    DEBUG(f"\n\nProcessing email: {this_email.subject}\n\n")
                    md_path, md_relative = assemble_journal_path(this_email.datetime_received, "Emails", this_email.subject, ".md")
                    tts_path, tts_relative = assemble_journal_path(this_email.datetime_received, "Emails", this_email.subject, ".wav")
                    if summarize_emails:
                        email_content = f'At {this_email.datetime_received}, {this_email.sender} sent an email with the subject line "{this_email.subject}". The email in its entirety reads: \n\n{this_email.body}\n"'
                        if this_email.attachments:
                            attachment_texts = await extract_attachments(this_email.attachments)
                            email_content += "\n—--\n" + "\n—--\n".join([f"Attachment: {text}" for text in attachment_texts])

                        summary = await summarize.summarize_text(email_content)
                        await tts.local_tts(text_content = summary, speed = 1.1, voice = DEFAULT_VOICE, podcast = podcast, output_path = tts_path)
                       
                        if podcast:
                            if PODCAST_DIR.exists():
                                tts.copy_to_podcast_dir(tts_path)
                            else:
                                ERR(f"PODCAST_DIR does not exist: {PODCAST_DIR}")

                        save_email_as_markdown(this_email, summary, md_path, tts_relative)
                    else:
                        save_email_as_markdown(this_email, None, md_path, None)

                    if auto_respond and should_auto_respond(this_email):
                        DEBUG(f"Auto-responding to {this_email.subject}")
                        auto_response_subject = 'Auto-Response Re:' + this_email.subject
                        auto_response_body = await generate_auto_response_body(this_email)
                        DEBUG(f"Auto-response: {auto_response_body}")
                        await send_auto_response(this_email.sender, auto_response_subject, auto_response_body)
                    
                    inbox.mark_seen(uid)

            await asyncio.sleep(30)
        except Exception as e:
            ERR(f"An error occurred: {e}")
            await asyncio.sleep(30)


def save_email_as_markdown(email: EmailModel, summary: str, md_path: Path, tts_path: Path):
    '''
Saves an email as a markdown file in the specified directory.
Args:
    email (EmailModel): The email object containing email details.
    summary (str): The summary of the email.
    tts_path (str): The path to the text-to-speech audio file.
    '''
    
    # Sanitize filename to avoid issues with filesystems
    filename = f"{email.datetime_received.strftime('%Y%m%d%H%M%S')}_{email.subject.replace('/', '-')}.md".replace(':', '-').replace(' ', '_')

    summary = prefix_lines(summary, '> ')
    # Create the markdown content
    markdown_content = f'''---
date: {email.datetime_received.strftime('%Y-%m-%d')}
tags:
 - email
---
|     |     |     | 
| --: | :--: |  :--: | 
|  *received* | **{email.datetime_received.strftime('%B %d, %Y at %H:%M:%S %Z')}**    |    |
|  *from* | **[[{email.sender}]]**    |    |
|  *to* | {', '.join([f'**[[{recipient}]]**' for recipient in email.recipients])}   |    |
|  *subject* | **{email.subject}**    |    |
'''
    
    if summary:
        markdown_content += f'''
> [!summary]  Summary
>  {summary}
'''
        
    if tts_path:
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


AUTORESPONSE_SYS = "You are a helpful AI assistant that generates personalized auto-response messages to incoming emails."

async def generate_auto_response_body(e: EmailModel, response_style: str = "professional") -> str:
    age = dt_datetime.now(TZ) - e.datetime_received
    prompt = f'''
Please generate a personalized auto-response to the following email. The email is from {e.sender} and was sent {age} ago with the subject line "{e.subject}." You are auto-responding on behalf of {USER_FULLNAME}, who is described by the following short bio (strictly for your context -- do not recite this in the response): "{USER_BIO}." {USER_FULLNAME} is unable to respond himself, because {AUTORESPONSE_CONTEXT}. Everything from here to ~~//END//~~ is the email body.
{e.body}
~~//END//~~
Keep your auto-response {response_style} and to the point, but do aim to make it responsive specifically to the sender's inquiry.
    '''
    
    try:
        response = await llm.query_ollama(prompt, AUTORESPONSE_SYS, 400)
        return response
    except Exception as e:
        ERR(f"Error generating auto-response: {str(e)}")
        return "Thank you for your email. Unfortunately, an error occurred while generating the auto-response. We apologize for any inconvenience."

async def send_auto_response(to_email, subject, body):
    try:
        message = MIMEMultipart()
        message['From'] = IMAP.email # smtp_username
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        # DEBUG(f"Attempting to send auto_response to {to_email} concerning {subject}. We will use {IMAP.host}:{IMAP.smtp_port}, un: {IMAP.email}, pw: {IMAP.password}")

        try:   
            DEBUG(f"Initiating attempt to send auto-response via SMTP at {IMAP.host}:{IMAP.smtp_port}...")
            context = ssl._create_unverified_context()

            with SMTP_SSL(IMAP.host, IMAP.smtp_port, context=context) as server:
                server.login(IMAP.email, IMAP.password)
                DEBUG(f"Successfully logged in to {IMAP.host} at {IMAP.smtp_port} as {IMAP.email}. Attempting to send email now.")
                server.send_message(message)

            INFO(f"Auto-response sent to {to_email} concerning {subject}")

        except Exception as e:
            ERR(f"Failed to send auto-response email to {to_email}: {e}")
            raise e

    except Exception as e:
        ERR(f"Error in preparing/sending auto-response: {e}")
        raise e

def should_auto_respond(email: EmailModel) -> bool:
    def matches_list(item: str, email: EmailModel) -> bool:
        if '@' in item:
            if item in email.sender:
                return True
        else:
            if item.lower() in email.subject.lower() or item.lower() in email.body.lower():
                return True
        return False
    
    if AUTORESPONSE_WHITELIST:
        for item in AUTORESPONSE_WHITELIST:
            if matches_list(item, email):
                if AUTORESPONSE_BLACKLIST:
                    for blacklist_item in AUTORESPONSE_BLACKLIST:
                        if matches_list(blacklist_item, email):
                            return False
                return True
        return False
    else:
        if AUTORESPONSE_BLACKLIST:
            for item in AUTORESPONSE_BLACKLIST:
                if matches_list(item, email):
                    return False
        return True
    
@email.on_event("startup")
async def startup_event():
    asyncio.create_task(process_unread_emails())