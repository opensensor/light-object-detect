"""Opt-in Sentry error and trace capture; no request payloads or local variables."""
import os
from urllib.parse import urlsplit, urlunsplit

import sentry_sdk


def scrub_event(event, hint):
    request = event.get("request")
    if request:
        for key in ("data", "cookies", "headers", "query_string"):
            request.pop(key, None)
        if request.get("url"):
            parts = urlsplit(request["url"])
            request["url"] = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    event.pop("user", None)
    event.pop("extra", None)
    event["breadcrumbs"] = {"values": []}
    return event


def init_sentry() -> None:
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn or os.environ.get("SENTRY_ENABLED", "true").lower() == "false":
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
        release=os.environ.get("SENTRY_RELEASE") or None,
        traces_sample_rate=0.1,
        data_collection={
            "user_info": False,
            "cookies": {"mode": "off"},
            "http_headers": {"request": {"mode": "off"}},
            "http_bodies": [],
            "url_query_params": {"mode": "off"},
            "database_query_data": False,
            "queues": False,
            "stack_frame_variables": False,
            "gen_ai": {"inputs": False, "outputs": False},
            "graphql": {"document": False, "variables": False},
        },
        before_send=scrub_event,
        before_send_transaction=scrub_event,
    )
