"""HTTP endpoints for one time code sign in, mounted next to /auth:

    GET  /auth/methods      {"methods": ["pin", "otp"], "otp_available": bool}
    POST /auth/otp/request  {phone, language?}
                            -> {challenge_id, masked_destination, channel,
                                expires_in, resend_in}
    POST /auth/otp/verify   {challenge_id, code}
                            -> {display_name, token, expires_at}
                               + httpOnly cookie user_session (SameSite=Lax),
                               exactly like POST /auth/verify

Failures (the page decides what to show from the status and "reason"):
    422 phone_format   not a Sri Lankan landline (same rule and message as
                       /auth/verify; decided before any lookup, so it says
                       nothing about who is registered)
    422                missing or malformed fields
    401 invalid        wrong, expired, used or unknown code: ONE answer
    423 locked         too many wrong codes on this challenge (Retry-After)
    429 rate_limited   hourly limit or too many verification attempts
    429 cooldown       a code was requested a moment ago (Retry-After)
    503 unavailable    no usable sender (OTP_SENDER not configured)
    404 disabled       "otp" is not in AUTH_METHODS

A request for a number that is unknown, or has no contact on record, gets
the same 200 answer as a customer; see service.py for how.

Handlers are plain def: the database is blocking, and FastAPI runs plain
def handlers on its thread pool instead of the event loop.
"""

from typing import Literal, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from auth.dependencies import USER_COOKIE, client_ip
from auth.routes import request_language
from auth.phone import InvalidPhoneNumber
from auth.tokens import AUDIENCE_USER, issue_token
from i18n import t
from otp import service

router = APIRouter(prefix="/auth", tags=["auth"])

_NO_STORE = {"Cache-Control": "no-store"}
INVALID_MESSAGE = "The code is incorrect or has expired."  # English default; the answer uses otp.bad_code


class OtpRequestBody(BaseModel):
    phone: str = Field(..., max_length=40)
    language: Optional[Literal["en", "si", "ta"]] = None


class OtpVerifyBody(BaseModel):
    challenge_id: str = Field(..., min_length=1, max_length=64)
    code: str = Field(..., min_length=1, max_length=12)


def _failure(error: service.OtpError, lang: str = "en") -> JSONResponse:
    retry = str(error.retry_after or 60)
    if error.reason == "locked":
        return JSONResponse(
            status_code=423,
            content={
                "detail": t("otp.locked", lang),
                "reason": "locked",
                "retry_after": error.retry_after,
            },
            headers={"Retry-After": retry, **_NO_STORE},
        )
    if error.reason in ("rate_limited", "cooldown"):
        detail = t("otp.too_soon", lang) if error.reason == "cooldown" else t("otp.rate_limited", lang)
        return JSONResponse(
            status_code=429,
            content={"detail": detail, "reason": error.reason, "retry_after": error.retry_after},
            headers={"Retry-After": retry, **_NO_STORE},
        )
    if error.reason == "unavailable":
        return JSONResponse(
            status_code=503,
            content={"detail": t("otp.unavailable", lang), "reason": "unavailable"},
            headers=_NO_STORE,
        )
    return JSONResponse(status_code=401, content={"detail": t("otp.bad_code", lang), "reason": "invalid"}, headers=_NO_STORE)


def _disabled(lang: str = "en") -> JSONResponse:
    return JSONResponse(
        status_code=404, content={"detail": t("otp.disabled", lang), "reason": "disabled"}, headers=_NO_STORE
    )


@router.get("/methods")
def methods():
    return JSONResponse(
        content={"methods": service.known_methods(), "otp_available": service.otp_available()},
        headers=_NO_STORE,
    )


@router.post("/otp/request")
def otp_request(body: OtpRequestBody, request: Request):
    lang = request_language(request, body.language)
    if not service.otp_enabled():
        return _disabled(lang)
    try:
        info = service.request_otp(body.phone, client_ip(request), body.language)
    except InvalidPhoneNumber as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc), "reason": "phone_format"})
    except service.OtpError as error:
        return _failure(error, lang)
    return JSONResponse(
        content={
            "challenge_id": info.challenge_id,
            "masked_destination": info.masked_destination,
            "channel": info.channel,
            "expires_in": info.expires_in,
            "resend_in": info.resend_in,
        },
        headers=_NO_STORE,
    )


@router.post("/otp/verify")
def otp_verify(body: OtpVerifyBody, request: Request):
    lang = request_language(request)
    if not service.otp_enabled():
        return _disabled(lang)
    try:
        user = service.verify_otp(body.challenge_id, body.code, client_ip(request))
    except service.OtpError as error:
        return _failure(error, lang)
    # Same token, audience, lifetime and cookie as POST /auth/verify, plus the
    # "otp" authentication method reference so verified customer tools unlock.
    token, expires_at = issue_token(
        str(user.id), AUDIENCE_USER, config.USER_SESSION_MINUTES, extra={"amr": ["otp"]}
    )
    response = JSONResponse(
        content={"display_name": user.display_name, "token": token, "expires_at": expires_at}, headers=_NO_STORE
    )
    response.set_cookie(
        key=USER_COOKIE,
        value=token,
        max_age=config.USER_SESSION_MINUTES * 60,
        path="/",
        httponly=True,
        secure=config.COOKIE_SECURE,
        samesite="lax",
    )
    return response
