.PHONY: dist

train:
	pipenv run python3 wincast/train.py

format:
	pipenv run python3 wincast/format.py

dist:
	pipenv run python3 setup.py sdist

pypi:
	pipenv run python3 setup.py sdist upload -r pypi
