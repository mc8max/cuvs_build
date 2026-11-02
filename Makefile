.DEFAULT_GOAL := help
.NOTPARALLEL:
PYTHON ?= python3
CONFIG ?= config/build.json
DRIVER = $(PYTHON) scripts/workflow.py --config "$(CONFIG)"
DOCKER_ARCHITECTURES ?=
DOCKER_JOBS ?=
DOCKER_DETACH ?= 0

.PHONY: help inventory doctor setup configure build docker-build package publish test benchmark fetch cycle check ssh smoke
help:
	@printf '%s\n' 'cuVS local build / RunPod execution' \
	  '  make inventory  Collect remote hardware/software into config/runpod.json' \
	  '  make doctor     Check existing local tools and inventory compatibility' \
	  '  make setup      Inventory, prepare local environment, configure cuVS' \
	  '  make smoke      Compile a tiny CUDA kernel locally and execute on RunPod' \
	  '  make build      Configure and compile the selected targets locally' \
	  '  make docker-build  Compile the configured targets in the Linux amd64 image' \
	  '  make publish    Build, bundle runtime libraries, upload a release via SSH' \
	  '  make test       Run tests on the last published release' \
	  '  make benchmark  Run a synthetic brute-force ANN benchmark remotely' \
	  '  make fetch      Download remote logs and results' \
	  '  make cycle      Publish, test, benchmark, fetch (fetch even after failure)' \
	  '  make check      Run local workflow checks (no GPU/SSH required)' \
	  '  make ssh        Open the configured SSH connection' \
	  'Set the local cuVS path in .env; edit config/build.json for build options.'
inventory:
	$(DRIVER) inventory
doctor:
	$(DRIVER) doctor
smoke:
	$(DRIVER) smoke
setup: inventory
	$(DRIVER) setup
configure:
	$(DRIVER) configure
build:
	$(DRIVER) build
docker-build:
	$(PYTHON) scripts/docker_build.py --config "$(CONFIG)" $(if $(DOCKER_JOBS),--jobs "$(DOCKER_JOBS)",) $(if $(DOCKER_ARCHITECTURES),--architectures $(DOCKER_ARCHITECTURES),) $(if $(filter 1,$(DOCKER_DETACH)),--detach,)
package: build
	$(DRIVER) package
publish: package
	$(DRIVER) publish
test:
	$(DRIVER) test
benchmark:
	$(DRIVER) benchmark
fetch:
	$(DRIVER) fetch
cycle: publish
	$(DRIVER) cycle
check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m unittest discover -s scripts -p 'test_*.py' -v
	$(PYTHON) -m py_compile scripts/*.py
	bash -n scripts/remote-run.sh
	bash -n docker/run-build.sh
ssh:
	$(DRIVER) ssh
