"""Notification agent for surfaced opportunities."""

from __future__ import annotations

import http.client
import os
import urllib.parse

from deal_finder.agents.base import Agent
from deal_finder.schemas import Opportunity


class MessagingAgent(Agent):
    """Send high-confidence deal alerts through configured channels."""

    name = "Messaging Agent"

    def format_alert(self, opportunity: Opportunity) -> str:
        return (
            f"Deal Alert: price=${opportunity.deal.price:.2f}, "
            f"estimate=${opportunity.estimate:.2f}, "
            f"discount=${opportunity.discount:.2f}. "
            f"{opportunity.deal.product_description[:120]} "
            f"{opportunity.deal.url}"
        )

    def alert(self, opportunity: Opportunity) -> None:
        text = self.format_alert(opportunity)
        if os.getenv("PUSHOVER_USER") and os.getenv("PUSHOVER_TOKEN"):
            self.push(text)
        if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
            self.text(text)
        self.log("notification workflow completed")

    def push(self, text: str) -> None:
        self.log("sending Pushover notification")
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request(
            "POST",
            "/1/messages.json",
            urllib.parse.urlencode(
                {
                    "token": os.environ["PUSHOVER_TOKEN"],
                    "user": os.environ["PUSHOVER_USER"],
                    "message": text,
                    "sound": "cashregister",
                }
            ),
            {"Content-type": "application/x-www-form-urlencoded"},
        )
        conn.getresponse()

    def text(self, text: str) -> None:
        self.log("sending Twilio SMS notification")
        from twilio.rest import Client

        client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
        client.messages.create(
            from_=os.environ["TWILIO_FROM"],
            body=text,
            to=os.environ["MY_PHONE_NUMBER"],
        )

