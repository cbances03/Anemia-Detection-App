import asyncio

from app import main


def test_loader_runs_once_per_application_lifespan(monkeypatch):
    real_loader = main.load_model_bundle
    calls = 0

    def counting_loader():
        nonlocal calls
        calls += 1
        return real_loader()

    monkeypatch.setattr(main, "load_model_bundle", counting_loader)

    async def run_lifespan():
        async with main.app.router.lifespan_context(main.app):
            assert main.app.state.model_bundle.model_version == "1.0.0"
            # Simulate several requests reading the already-loaded resource.
            assert main.app.state.model_bundle.model is not None
            assert main.app.state.model_bundle.model is not None

    asyncio.run(run_lifespan())

    assert calls == 1
    assert main.app.state.model_bundle is None
