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
	@$(PIP) --version || (echo "No pip distribution detected"; exit 1)


install:
	@echo
	@echo "Installing ColECM"
	@echo
	@$(PYTHON) make.py install $(NAME) || (echo "Installation failed"; exit 1)
	@$(PIP) install -r requirements.txt 


test:
	@echo
	@echo "Running unit tests"
	@echo
	@pytest tests/ -v


uninstall:
	@$(PYTHON) make.py uninstall $(NAME) || (echo "Uninstallation failed"; exit 1)
		

clean:
	@rm -f -r tests/__pycache__
	@rm -f -r src/__pycache__
	@rm -f src/*.pyc
	@rm -f -r .cache/
	@rm -f -r .DS_Store/
	@rm -f -r .pytest_cache/
