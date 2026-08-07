import asyncio
import json
from types import SimpleNamespace

from app import main
from app.core import model_loader


async def asgi_get(path: str) -> tuple[int, dict]:
    messages = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("test", 123),
        "server": ("testserver", 80),
        "root_path": "",
        "app": main.app,
    }
    await main.app(scope, receive, send)

    response_start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return response_start["status"], json.loads(response_body)


def test_model_info_returns_loaded_model_and_does_not_reload(monkeypatch):
    real_joblib_load = model_loader.joblib.load
    load_calls = 0

    def counting_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return real_joblib_load(*args, **kwargs)

    monkeypatch.setattr(model_loader.joblib, "load", counting_load)

    async def exercise_endpoint():
        async with main.app.router.lifespan_context(main.app):
            responses = [await asgi_get("/model/info") for _ in range(3)]
            for response_status, body in responses:
                assert response_status == 200
                assert body["model_version"] == "1.0.0"
                assert body["model_name"] == "kNN_SMOTE"
                assert body["threshold"] == 0.40
                assert body["status"] == "production_candidate"

    asyncio.run(exercise_endpoint())
    assert load_calls == 1


def test_model_info_uses_existing_bundle(monkeypatch):
    existing_bundle = SimpleNamespace(
        model_name="bundle-model",
        model_version="bundle-version",
        threshold=0.25,
        metadata={
            "model": {"status": "bundle-status"},
            "decision": {"threshold": 0.25},
        },
        positive_class="positive",
        positive_class_index=1,
        feature_count=9,
    )
    main.app.state.model_bundle = existing_bundle
    monkeypatch.setattr(
        model_loader.joblib,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("joblib.load must not run in the endpoint")
        ),
    )

    response_status, body = asyncio.run(asgi_get("/model/info"))

    assert response_status == 200
    assert body["model_name"] == "bundle-model"
    assert body["model_version"] == "bundle-version"
    assert body["threshold"] == 0.25
    main.app.state.model_bundle = None


def test_model_info_returns_503_when_bundle_is_unavailable():
    main.app.state.model_bundle = None

    response_status, body = asyncio.run(asgi_get("/model/info"))

    assert response_status == 503
    assert body == {"detail": "Model is not available"}
