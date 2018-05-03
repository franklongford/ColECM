PYTHON=python
PIP=pip

NAME=ColECM

init: check install test


check:
	@echo
	@echo "Checking default python version:"
	@$(PYTHON) --version || (echo "No python distribution detected"; exit 1)
	@echo
	@echo "Checking default pip version:"
	@$(PYTHON) -m $(PIP) --version || (echo "No pip distribution detected"; exit 1)


install:
	@echo
	@echo "Installing ColECM: creating binary"
	@echo
	@$(PYTHON) install.py $(NAME) || (echo "Installation failed"; exit 1)
	@chmod +x $(NAME)
	@$(PYTHON) -m $(PIP) install -r requirements.txt


test:
	@echo
	@echo "Running unit tests"
	@echo
	@pytest tests/ -v


uninstall:
	@rm -f ColECM


clean:
	@rm -f -r tests/__pycache__
	@rm -f -r src/__pycache__
	@rm -f src/*.pyc
	@rm -f -r .cache/
	@rm -f -r .DS_Store/
	@rm -f -r .pytest_cache/
