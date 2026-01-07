from investigator.story_loader import load_story_messages

def test_load_story_messages_returns_messages():
    msgs = load_story_messages("whodunit_sydney.xml")
    assert len(msgs) > 50  
    assert msgs[0].id
    assert msgs[0].ts
    assert msgs[0].body

def test_sender_name_resolved():
    msgs = load_story_messages("whodunit_sydney.xml")
    assert any(m.sender_name for m in msgs)
