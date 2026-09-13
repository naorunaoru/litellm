from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


def test_chatgpt_normalizes_system_messages_to_developer_without_mutating_input():
    config = ChatGPTConfig()
    messages = [
        {"role": "system", "content": "Follow the policy", "name": "policy"},
        {"role": "user", "content": "Hello"},
    ]

    transformed = config._transform_messages(messages=messages, model="gpt-6-astra")

    assert transformed == [
        {"role": "developer", "content": "Follow the policy", "name": "policy"},
        {"role": "user", "content": "Hello"},
    ]
    assert messages[0]["role"] == "system"