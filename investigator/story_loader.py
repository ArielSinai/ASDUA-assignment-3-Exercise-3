from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import xml.etree.ElementTree as ET


NS = {"w": "urn:whodunit:sms:1"}


@dataclass(frozen=True)
class StoryMessage:
    id: str
    ts: str
    sender_id: Optional[str]
    sender_name: Optional[str]
    receiver_id: Optional[str]
    receiver_name: Optional[str]
    body: str


def load_story_messages(xml_path: str) -> list[StoryMessage]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # participants map
    participants = root.find("w:participants", NS)
    people = {}
    if participants is not None:
        for p in participants.findall("w:person", NS):
            pid = p.attrib.get("id")
            name = p.attrib.get("name")
            if pid:
                people[pid] = name

    def ref_id(el: Optional[ET.Element]) -> Optional[str]:
        if el is None:
            return None
        return el.attrib.get("ref")

    messages: list[StoryMessage] = []
    for m in root.findall(".//w:message", NS):
        mid = m.attrib.get("id", "")
        ts = m.attrib.get("ts", "")

        sender_el = m.find("w:sender", NS)
        receiver_el = m.find("w:receiver", NS)
        body_el = m.find("w:body", NS)

        sid = ref_id(sender_el)
        rid = ref_id(receiver_el)
        body = (body_el.text or "").strip() if body_el is not None else ""

        messages.append(
            StoryMessage(
                id=mid,
                ts=ts,
                sender_id=sid,
                sender_name=people.get(sid),
                receiver_id=rid,
                receiver_name=people.get(rid),
                body=body,
            )
        )

    return messages
