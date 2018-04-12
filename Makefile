PYTHON=python3
PIP=pip3

init:
	$(PYTHON) -m venv ColECM_venv
	ColECM_venv/bin/activate
	$(PIP) install -r requirements.txt

test:
	pytest tests/

clean:
	rm -r ColECM_venv
