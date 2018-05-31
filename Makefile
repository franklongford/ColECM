PYTHON=python
PIP=pip

BIN=~/.conda/envs/python_3/bin/ "Enter binary folder here"

NAME=ColECM
NAME_MPI=ColECM_mpi


init: check install test


check:
	@echo $(PYTHON)
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
	@$(PIP) install -r requirements.txt
	@$(PYTHON) make.py install $(NAME) $(BIN) || (echo "Installation failed"; exit 1)

install_mpi:
	@echo
	@echo "Installing ColECM"
	@echo
	@$(PIP) install -r requirements.txt
	@$(PYTHON) make.py install_mpi $(NAME_MPI) $(BIN) || (echo "Installation failed"; exit 1)

test:
	@echo
	@echo "Running unit tests"
	@echo
	@pytest tests/ -v -l


uninstall:
	@$(PYTHON) make.py uninstall $(NAME) $(BIN) || (echo "Uninstallation failed"; exit 1)
		

uninstall_mpi:
	@$(PYTHON) make.py uninstall_mpi $(NAME) $(BIN) || (echo "Uninstallation failed"; exit 1)

clean:
	@rm -f -r bin
	@rm -f -r tests/__pycache__
	@rm -f -r src/__pycache__
	@rm -f src/*.pyc
	@rm -f -r .cache/
	@rm -f -r .DS_Store/
	@rm -f -r .pytest_cache/
