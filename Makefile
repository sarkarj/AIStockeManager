.PHONY: test drl-verify green llm-smoke run-streamlit run-graph-api run-prewarm run-mcp drl-trace hub-verify

test:
	pytest -q

drl-verify:
	python3 scripts/verify_drl_integrity.py

green:
	$(MAKE) test
	$(MAKE) drl-verify

llm-smoke:
	python3 scripts/llm_smoke_check.py

hub-verify:
	python3 scripts/hub_verify.py

run-streamlit:
	streamlit run app/ui/streamlit_app.py

run-graph-api:
	python3 -m uvicorn app.api.graph_api:app --host 0.0.0.0 --port 8701

run-prewarm:
	python3 scripts/prewarm_worker.py

run-mcp:
	python3 -m app.core.mcp.server

drl-trace:
	@if [ -z "$(FIXTURE)" ]; then \
		echo "Usage: make drl-trace FIXTURE=F02_OVERSOLD_BEAR_ADX_STRONG_REDUCE_LOWCONF"; \
		exit 1; \
	fi
	python3 scripts/print_drl_trace.py --fixture-id $(FIXTURE)
