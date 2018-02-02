.PHONY: dist

dist:
	python3 setup.py sdist

pypi:
	python3 setup.py sdist upload -r pypi
