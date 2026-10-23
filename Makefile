.PHONY: test lint smoke api dashboard
test:
	python -m pytest
lint:
	python -m ruff check .
smoke:
	python scripts/smoke_test.py
api:
	uvicorn api.main:app --reload
dashboard:
	streamlit run app/dashboard.py
