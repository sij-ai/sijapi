from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime

class AutoResponder(BaseModel):
    name: str
    style: str
    context: str
    whitelist: List[str]
    blacklist: List[str]
    img_gen_prompt: Optional[str] = None

class IMAPConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    encryption: str = None

class SMTPConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    encryption: str = None

class EmailAccount(BaseModel):
    name: str
    fullname: Optional[str]
    bio: Optional[str]
    imap: IMAPConfig
    smtp: SMTPConfig
    autoresponders: Optional[List[AutoResponder]]

class EmailContact(BaseModel):
    email: str
    name: str

class IncomingEmail(BaseModel):
    sender: str
    recipients: List[EmailContact]
    datetime_received: datetime
    subject: str
    body: str
    attachments: Optional[List[Any]] = None