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
import ssl
from smtplib import SMTP_SSL
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime as dt_datetime
from pydantic import BaseModel
from typing import List, Optional, Any
import yaml
from typing import List, Dict, Optional
from pydantic import BaseModel

from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import PODCAST_DIR, DEFAULT_VOICE, TZ, EMAIL_ACCOUNTS, EmailAccount, IMAPConfig, SMTPConfig
from sijapi.routers import summarize, tts, llm, sd
from sijapi.utilities import clean_text, assemble_journal_path, localize_datetime, extract_text, prefix_lines
from sijapi.classes import EmailAccount, IncomingEmail, EmailContact


email = APIRouter(tags=["private"])

def get_account_by_email(email: str) -> Optional[EmailAccount]:
    for account in EMAIL_ACCOUNTS:
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
                'IMG_GEN_PROMPT': profile.img_gen_prompt,
                'USER_BIO': account.bio
            })

    return matching_profiles


async def generate_auto_response_body(e: IncomingEmail, profile: Dict) -> str:
    age = dt_datetime.now(TZ) - e.datetime_received
    prompt = f'''
Please generate a personalized auto-response to the following email. The email is from {e.sender} and was sent {age} ago with the subject line "{e.subject}." You are auto-responding on behalf of {profile['USER_FULLNAME']}, who is described by the following short bio (strictly for your context -- do not recite this in the response): "{profile['USER_BIO']}." {profile['USER_FULLNAME']} is unable to respond personally, because {profile['AUTORESPONSE_CONTEXT']}. Everything from here to ~~//END//~~ is the email body.
{e.body}
~~//END//~~
Keep your auto-response {profile['RESPONSE_STYLE']} and to the point, but do aim to make it responsive specifically to the sender's inquiry.
    '''
    
    try:
        response = await llm.query_ollama(prompt, 400)
        return response
    except Exception as e:
        ERR(f"Error generating auto-response: {str(e)}")
        return "Thank you for your email. Unfortunately, an error occurred while generating the auto-response. We apologize for any inconvenience."



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


async def process_unread_emails(summarize_emails: bool = True, podcast: bool = True):
    while True:
        for account in EMAIL_ACCOUNTS:
            DEBUG(f"Connecting to {account.name} to check for unread emails...")
            try:
                with get_imap_connection(account) as inbox:
                    DEBUG(f"Connected to {account.name}, checking for unread emails now...")
                    unread_messages = inbox.messages(unread=True)
                    for uid, message in unread_messages:
                        recipients = [EmailContact(email=recipient['email'], name=recipient.get('name', '')) for recipient in message.sent_to]
                        this_email = IncomingEmail(
                            sender=message.sent_from[0]['email'],
                            datetime_received=localize_datetime(message.date),
                            recipients=recipients,
                            subject=message.subject,
                            body=clean_email_content(message.body['html'][0]) if message.body['html'] else clean_email_content(message.body['plain'][0]) or "",
                            attachments=message.attachments
                        )

                        DEBUG(f"\n\nProcessing email for account {account.name}: {this_email.subject}\n\n")

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
                            DEBUG(f"Email '{this_email.subject}' saved to {md_relative}.")
                        else:
                            save_email_as_markdown(this_email, None, md_path, None)

                        matching_profiles = get_matching_autoresponders(this_email, account)

                        for profile in matching_profiles:
                            DEBUG(f"Auto-responding to {this_email.subject} with profile: {profile['USER_FULLNAME']}")
                            auto_response_subject = f"Auto-Response Re: {this_email.subject}"
                            auto_response_body = await generate_auto_response_body(this_email, profile)
                            DEBUG(f"Auto-response: {auto_response_body}")
                            await send_auto_response(this_email.sender, auto_response_subject, auto_response_body, profile, account)
                        
                        inbox.mark_seen(uid)

                await asyncio.sleep(30)
            except Exception as e:
                ERR(f"An error occurred for account {account.name}: {e}")
                await asyncio.sleep(30)



def save_email_as_markdown(email: IncomingEmail, summary: str, md_path: Path, tts_path: Path):
    '''
Saves an email as a markdown file in the specified directory.
Args:
    email (IncomingEmail): The email object containing email details.
    summary (str): The summary of the email.
    tts_path (str): The path to the text-to-speech audio file.
    '''
    DEBUG(f"Saving email to {md_path}...")
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


async def send_auto_response(to_email, subject, body, profile, account):
    DEBUG(f"Sending auto response to {to_email}...")
    try:
        message = MIMEMultipart()
        message['From'] = account.smtp.username
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        if profile['IMG_GEN_PROMPT']:
            jpg_path = sd.workflow(profile['IMG_GEN_PROMPT'], earlyout=False, downscale_to_fit=True)
            if jpg_path and os.path.exists(jpg_path):
                with open(jpg_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read(), name=os.path.basename(jpg_path))
                    message.attach(img)

        context = ssl._create_unverified_context()
        with SMTP_SSL(account.smtp.host, account.smtp.port, context=context) as server:
            server.login(account.smtp.username, account.smtp.password)
            server.send_message(message)

        INFO(f"Auto-response sent to {to_email} concerning {subject} from account {account.name}")

    except Exception as e:
        ERR(f"Error in preparing/sending auto-response from account {account.name}: {e}")
        raise e



    
@email.on_event("startup")
async def startup_event():
    asyncio.create_task(process_unread_emails())





    ####



