"""Transactional email. Uses SMTP when configured, otherwise logs the message (development)."""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from .config import settings

log = logging.getLogger("aerosculpt.email")


def send(to: str, subject: str, body: str) -> bool:
    if not settings.smtp_host:
        log.warning("SMTP not configured; email to %s:\nSubject: %s\n\n%s", to, subject, body)
        return False
    msg = EmailMessage()
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as s:
            s.starttls(context=ssl.create_default_context())
            if settings.smtp_user:
                s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
        return True
    except (smtplib.SMTPException, OSError) as e:
        log.error("email to %s failed: %s", to, e)
        return False
