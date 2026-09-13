from litellm.llms.chatgpt.common_utils import (
    derive_chatgpt_session_id,
    get_chatgpt_default_headers,
    should_derive_chatgpt_session_id,
)


def test_chatgpt_session_header_uses_codex_wire_name():
    headers = get_chatgpt_default_headers("access-token", "account-id", "session-123")

    assert headers["session-id"] == "session-123"
    assert "session_id" not in headers


def test_chatgpt_session_derivation_is_securely_enabled_by_default():
    assert should_derive_chatgpt_session_id({}) is True
    assert should_derive_chatgpt_session_id({"chatgpt_derive_session_id": False}) is False


def test_derived_session_id_is_stable_and_tenant_scoped(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    first_params = {"litellm_metadata": {"user_api_key_hash": "tenant-a"}}
    second_params = {"litellm_metadata": {"user_api_key_hash": "tenant-b"}}
    request_input = [{"role": "user", "content": "shared prefix"}]

    first_id = derive_chatgpt_session_id(first_params, "instructions", request_input, "gpt-5.6-sol")
    repeated_id = derive_chatgpt_session_id(first_params, "instructions", request_input, "gpt-5.6-sol")
    second_id = derive_chatgpt_session_id(second_params, "instructions", request_input, "gpt-5.6-sol")

    assert first_id == repeated_id
    assert first_id != second_id
    assert first_id is not None
    assert first_id.startswith("litellm-derived-v1-")


def test_derived_chatgpt_session_id_fails_closed_without_tenant_or_salt(monkeypatch):
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    params = {"litellm_metadata": {"user_api_key_hash": "tenant-a"}}
    request_input = [{"role": "user", "content": "shared prefix"}]

    assert derive_chatgpt_session_id(params, "instructions", request_input, "gpt-5.6-sol") is None

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt")
    assert derive_chatgpt_session_id({}, "instructions", request_input, "gpt-5.6-sol") is None
