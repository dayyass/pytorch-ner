all:
	python -m pytorch_ner --path_to_config config.yaml
coverage:
	coverage run -m unittest discover && coverage report -m
docker_build:
	docker image build -t pytorch-ner .
docker_run:
	docker container run -it pytorch-ner
pypi_packages:
	pip install --upgrade build twine
pypi_build:
	python -m build
pypi_twine:
	python -m twine upload --repository testpypi dist/*
pypi_clean:
	rm -rf dist pytorch_ner.egg-info
clean:
	rm -rf models/*
