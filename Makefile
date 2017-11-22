all: docs

.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "*.npy" -delete
	find . -name "*.csv" -delete

.PHONY: docs
docs:
	sphinx-apidoc -f -o docs/source/ dog_classifier/ && pushd docs && make html && popd
