from __future__ import annotations
import threading
from pathlib import Path
import smtplib
from email.message import EmailMessage
from loguru import logger
from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND

# Shared lock for frame access
lock = threading.Lock()
SNAP_DIR = Path(__file__).resolve().parent.parent / "snapshots"
SNAP_DIR.mkdir(exist_ok=True)

def require_roles(request: Request, roles: list[str]):
    user = request.session.get("user")
    if not user or user.get("role") not in roles:
        return RedirectResponse("/login", status_code=HTTP_302_FOUND)
    return user

def send_email(subject: str, message: str, to_list: list[str], image: bytes | None = None, cfg: dict | None = None, attachment: bytes | None = None, attachment_name: str | None = None, attachment_type: str | None = None) -> None:
    """Send an email with optional JPEG or arbitrary attachment."""
    if not cfg:
        return
    host = cfg.get("smtp_host")
    if not host:
        return
    to_addrs = [a.strip() for a in to_list if a.strip()]
    msg = EmailMessage()
    msg["From"] = cfg.get("from_addr")
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg.set_content(message)
    if image:
        msg.add_attachment(image, maintype="image", subtype="jpeg", filename="alert.jpg")
    if attachment:
        maintype, subtype = attachment_type.split('/') if attachment_type else ("application", "octet-stream")
        msg.add_attachment(attachment, maintype=maintype, subtype=subtype, filename=attachment_name or "attachment.bin")
    try:
        with smtplib.SMTP(host, cfg.get("smtp_port", 587)) as s:
            if cfg.get("use_tls", True):
                s.starttls()
            if cfg.get("smtp_user"):
                s.login(cfg.get("smtp_user"), cfg.get("smtp_pass", ""))
            s.send_message(msg)
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
