# Makefile for common tasks

.PHONY: progress

progress:
	python -u tools/show_progress.py
	@echo "Generated progress.html"
