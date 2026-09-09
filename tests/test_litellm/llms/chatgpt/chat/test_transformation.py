from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


def test_chatgpt_normalizes_system_messages_to_developer():
    config = ChatGPTConfig()

    transformed = config._transform_messages(
        messages=[
            {"role": "system", "content": "Follow the policy"},
            {"role": "user", "content": "Hello"},
        ],
        model="gpt-6-astra",
    )

    assert transformed == [
        {"role": "developer", "content": "Follow the policy"},
        {"role": "user", "content": "Hello"},
    ]
