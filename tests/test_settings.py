from src.paperlens.settings import Settings


def test_local_env_file_overrides_base_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    (tmp_path / ".env").write_text("LLM_MODEL=base-model\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("LLM_MODEL=local-model\n", encoding="utf-8")

    assert Settings().llm_model == "local-model"
