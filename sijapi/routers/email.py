'''
Uses IMAP and SMTP login credentials to monitor an inbox and summarize incoming emails that match certain criteria and save the Text-To-Speech converted summaries into a specified "podcast" folder. 
'''
from fastapi import APIRouter
import asyncio
import aiofiles
from imbox import Imbox
from bs4 import BeautifulSoup
import os
from pathlib import Path
from shutil import move
import tempfile
import re
import traceback
from smtplib import SMTP_SSL, SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import ssl
import yaml
from typing import List, Dict, Optional, Set
from datetime import datetime as dt_datetime
from sijapi import L, PODCAST_DIR, DEFAULT_VOICE, EMAIL_CONFIG, EMAIL_LOGS
from sijapi.routers import img, loc, tts, llm
from sijapi.utilities import clean_text, assemble_journal_path, extract_text, prefix_lines
from sijapi.classes import EmailAccount, IMAPConfig, SMTPConfig, IncomingEmail, EmailContact, AutoResponder
from sijapi.classes import EmailAccount

email = APIRouter(tags=["private"])


def load_email_accounts(yaml_path: str) -> List[EmailAccount]:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return [EmailAccount(**account) for account in config['accounts']]


def get_imap_connection(account: EmailAccount):
    return Imbox(account.imap.host,
        username=account.imap.username,
        password=account.imap.password,
        port=account.imap.port,
        ssl=account.imap.encryption == 'SSL',
        starttls=account.imap.encryption == 'STARTTLS')



def get_smtp_connection(autoresponder):
    # Create an SSL context that doesn't verify certificates
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    if autoresponder.smtp.encryption == 'SSL':
        try:
            L.DEBUG(f"Attempting SSL connection to {autoresponder.smtp.host}:{autoresponder.smtp.port}")
            return SMTP_SSL(autoresponder.smtp.host, autoresponder.smtp.port, context=context)
        except ssl.SSLError as e:
            L.ERR(f"SSL connection failed: {str(e)}")
            # If SSL fails, try TLS
            try:
                L.DEBUG(f"Attempting STARTTLS connection to {autoresponder.smtp.host}:{autoresponder.smtp.port}")
                smtp = SMTP(autoresponder.smtp.host, autoresponder.smtp.port)
                smtp.starttls(context=context)
                return smtp
            except Exception as e:
                L.ERR(f"STARTTLS connection failed: {str(e)}")
                raise
    elif autoresponder.smtp.encryption == 'STARTTLS':
        try:
            L.DEBUG(f"Attempting STARTTLS connection to {autoresponder.smtp.host}:{autoresponder.smtp.port}")
            smtp = SMTP(autoresponder.smtp.host, autoresponder.smtp.port)
            smtp.starttls(context=context)
            return smtp
        except Exception as e:
            L.ERR(f"STARTTLS connection failed: {str(e)}")
            raise
    else:
        try:
            L.DEBUG(f"Attempting unencrypted connection to {autoresponder.smtp.host}:{autoresponder.smtp.port}")
            return SMTP(autoresponder.smtp.host, autoresponder.smtp.port)
        except Exception as e:
            L.ERR(f"Unencrypted connection failed: {str(e)}")
            raise


async def send_response(to_email: str, subject: str, body: str, profile: AutoResponder, image_attachment: Path = None) -> bool:
    server = None
    try:
        message = MIMEMultipart()
        message['From'] = profile.smtp.username
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        if image_attachment and os.path.exists(image_attachment):
            with open(image_attachment, 'rb') as img_file:
                img = MIMEImage(img_file.read(), name=os.path.basename(image_attachment))
                message.attach(img)

        L.DEBUG(f"Sending auto-response to {to_email} concerning {subject} from account {profile.name}...")

        server = get_smtp_connection(profile.smtp)
        L.DEBUG(f"SMTP connection established: {type(server)}")
        server.login(profile.smtp.username, profile.smtp.password)
        server.send_message(message)

        L.INFO(f"Auto-response sent to {to_email} concerning {subject} from account {profile.name}!")
        return True

    except Exception as e:
        L.ERR(f"Error in preparing/sending auto-response from account {profile.name}: {str(e)}")
        L.ERR(f"SMTP details - Host: {profile.smtp.host}, Port: {profile.smtp.port}, Encryption: {profile.smtp.encryption}")
        L.ERR(traceback.format_exc())
        return False

    finally:
        if server:
            try:
                server.quit()
            except Exception as e:
                L.ERR(f"Error closing SMTP connection: {str(e)}")



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

async def process_account_archival(account: EmailAccount):
    summarized_log = EMAIL_LOGS / account.name / "summarized.txt"
    os.makedirs(summarized_log.parent, exist_ok = True)

    while True:
        try:
            processed_uids = await load_processed_uids(summarized_log)
            L.DEBUG(f"{len(processed_uids)} emails marked as already summarized are being ignored.")
            with get_imap_connection(account) as inbox:
                unread_messages = inbox.messages(unread=True)
                L.DEBUG(f"There are {len(unread_messages)} unread messages.")
                for uid, message in unread_messages:
                    uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                    if uid_str not in processed_uids:
                        recipients = [EmailContact(email=recipient['email'], name=recipient.get('name', '')) for recipient in message.sent_to]
                        localized_datetime = await loc.dt(message.date)
                        this_email = IncomingEmail(
                            sender=message.sent_from[0]['email'],
                            datetime_received=localized_datetime,
                            recipients=recipients,
                            subject=message.subject,
                            body=clean_email_content(message.body['html'][0]) if message.body['html'] else clean_email_content(message.body['plain'][0]) or "",
                            attachments=message.attachments 
                        )
                        md_path, md_relative = assemble_journal_path(this_email.datetime_received, "Emails", this_email.subject, ".md")
                        md_summary = await summarize_single_email(this_email, account.podcast) if account.summarize == True else None
                        md_content = await archive_single_email(this_email, md_summary)
                        save_success = await save_email(md_path, md_content)
                        if save_success:
                            await save_processed_uid(summarized_log, account.name, uid_str)
                            L.INFO(f"Summarized email: {uid_str}")
                        else:
                            L.WARN(f"Failed to summarize {this_email.subject}")
                    else:
                        L.DEBUG(f"Skipping {uid_str} because it was already processed.")
        except Exception as e:
            L.ERR(f"An error occurred during summarization for account {account.name}: {e}")
        
        await asyncio.sleep(account.refresh)

async def summarize_single_email(this_email: IncomingEmail, podcast: bool = False):
    tts_path, tts_relative = assemble_journal_path(this_email.datetime_received, "Emails", this_email.subject, ".wav")
    summary = ""
    email_content = f'At {this_email.datetime_received}, {this_email.sender} sent an email with the subject line "{this_email.subject}". The email in its entirety reads: \n\n{this_email.body}\n"'
    if this_email.attachments:
        attachment_texts = await extract_attachments(this_email.attachments)
        email_content += "\n—--\n" + "\n—--\n".join([f"Attachment: {text}" for text in attachment_texts])
    summary = await llm.summarize_text(email_content)
    await tts.local_tts(text_content = summary, speed = 1.1, voice = DEFAULT_VOICE, podcast = podcast, output_path = tts_path)
    md_summary = f'```ad.summary\n'
    md_summary += f'title: {this_email.subject}\n'
    md_summary += f'{summary}\n'
    md_summary += f'```\n\n'
    md_summary += f'![[{tts_relative}]]\n'# if tts_path.exists() else ''

    return md_summary

async def archive_single_email(this_email: IncomingEmail, summary: str = None):
    try:
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
            markdown_content += summary

        markdown_content += f'''
---
{this_email.body}
'''
        return markdown_content
    
    except Exception as e:
        L.ERR(f"Exception: {e}")
        return False
    
async def save_email(md_path, md_content):
    try:
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(md_content)

        L.DEBUG(f"Saved markdown to {md_path}")
        return True
    except Exception as e:
        L.ERR(f"Failed to save email: {e}")
        return False

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


async def process_account_autoresponding(account: EmailAccount):
    EMAIL_AUTORESPONSE_LOG = EMAIL_LOGS / account.name / "autoresponded.txt"
    os.makedirs(EMAIL_AUTORESPONSE_LOG.parent, exist_ok=True)

    while True:
        try:
            processed_uids = await load_processed_uids(EMAIL_AUTORESPONSE_LOG)
            L.DEBUG(f"{len(processed_uids)} emails marked as already responded to are being ignored.")

            with get_imap_connection(account) as inbox:
                unread_messages = inbox.messages(unread=True)
                L.DEBUG(f"There are {len(unread_messages)} unread messages.")

                for uid, message in unread_messages:
                    uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                    if uid_str not in processed_uids:
                        await autorespond_single_email(message, uid_str, account, EMAIL_AUTORESPONSE_LOG)
                    else:
                        L.DEBUG(f"Skipping {uid_str} because it was already processed.")

        except Exception as e:
            L.ERR(f"An error occurred during auto-responding for account {account.name}: {e}")
        
        await asyncio.sleep(account.refresh)

async def autorespond_single_email(message, uid_str: str, account: EmailAccount, log_file: Path):
    this_email = await create_incoming_email(message)
    L.DEBUG(f"Evaluating {this_email.subject} for autoresponse-worthiness...")
    
    matching_profiles = get_matching_autoresponders(this_email, account)
    L.DEBUG(f"Matching profiles: {matching_profiles}")

    for profile in matching_profiles:
        response_body = await generate_response(this_email, profile, account)
        if response_body:
            subject = f"Re: {this_email.subject}"
            # add back scene=profile.image_scene,  to workflow call
            jpg_path = await img.workflow(profile.image_prompt, earlyout=False, downscale_to_fit=True) if profile.image_prompt else None
            success = await send_response(this_email.sender, subject, response_body, profile, jpg_path)
            if success:
                L.WARN(f"Auto-responded to email: {this_email.subject}")
                await save_processed_uid(log_file, account.name, uid_str)
            else:
                L.WARN(f"Failed to send auto-response to {this_email.subject}")
        else:
            L.WARN(f"Unable to generate auto-response for {this_email.subject}")

async def generate_response(this_email: IncomingEmail, profile: AutoResponder, account: EmailAccount) -> Optional[str]:
    L.INFO(f"Generating auto-response to {this_email.subject} with profile: {profile.name}")

    now = await loc.dt(dt_datetime.now())
    then = await loc.dt(this_email.datetime_received)
    age = now - then
    usr_prompt = f'''
Generate a personalized auto-response to the following email:
From: {this_email.sender}
Sent: {age} ago
Subject: "{this_email.subject}"
Body: {this_email.body}
---
Respond on behalf of {account.fullname}, who is unable to respond personally because {profile.context}. Keep the response {profile.style} and to the point, but responsive to the sender's inquiry. Do not mention or recite this context information in your response.
    '''
    sys_prompt = f"You are an AI assistant helping {account.fullname} with email responses. {account.fullname} is described as: {account.bio}"
    
    try:
        response = await llm.query_ollama(usr_prompt, sys_prompt, profile.ollama_model, 400)
        L.DEBUG(f"query_ollama response: {response}")
        
        if isinstance(response, dict) and "message" in response and "content" in response["message"]:
            response = response["message"]["content"]
        
        return response + "\n\n"
        
    except Exception as e:
        L.ERR(f"Error generating auto-response: {str(e)}")
        return None



async def create_incoming_email(message) -> IncomingEmail:
    recipients = [EmailContact(email=recipient['email'], name=recipient.get('name', '')) for recipient in message.sent_to]
    localized_datetime = await loc.dt(message.date)
    return IncomingEmail(
        sender=message.sent_from[0]['email'],
        datetime_received=localized_datetime,
        recipients=recipients,
        subject=message.subject,
        body=clean_email_content(message.body['html'][0]) if message.body['html'] else clean_email_content(message.body['plain'][0]) or "",
        attachments=message.attachments
    )

async def load_processed_uids(filename: Path) -> Set[str]:
    if filename.exists():
        async with aiofiles.open(filename, 'r') as f:
            return set(line.strip().split(':')[-1] for line in await f.readlines())
    return set()

async def save_processed_uid(filename: Path, account_name: str, uid: str):
    async with aiofiles.open(filename, 'a') as f:
        await f.write(f"{account_name}:{uid}\n")


async def process_all_accounts():
    email_accounts = load_email_accounts(EMAIL_CONFIG)
    summarization_tasks = [asyncio.create_task(process_account_archival(account)) for account in email_accounts]
    autoresponding_tasks = [asyncio.create_task(process_account_autoresponding(account)) for account in email_accounts]
    await asyncio.gather(*summarization_tasks, *autoresponding_tasks)

@email.on_event("startup")
async def startup_event():
    await asyncio.sleep(5)
    asyncio.create_task(process_all_accounts())
