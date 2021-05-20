IMAGE_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
GIT_TAG ?= $(shell git log --oneline | head -n1 | awk '{print $$1}')
DOCKER_REGISTRY := mathematiguy
IMAGE := $(DOCKER_REGISTRY)/$(IMAGE_NAME)
HAS_DOCKER ?= $(shell which docker)
RUN ?= $(if $(HAS_DOCKER), docker run $(DOCKER_ARGS) --rm -it --ipc host --gpus all -v $$(pwd):/work -w /work -u $(UID):$(GID) $(IMAGE))
UID ?= $(shell id -u)
GID ?= $(shell id -g)
DOCKER_ARGS ?=
LOG_LEVEL ?= INFO

.PHONY: docker docker-push docker-pull enter enter-root

data: data/unzipped/nzherald_harvest.csv

data/unzipped/nzherald_harvest.csv: data/raw/nzherald_harvest.csv.gz
	cat $< | gzip -d > $@

prepare: data/train/train.txt
data/train/train.txt: scripts/prepare.py data/unzipped/nzherald_harvest.csv
	$(RUN) python3 $< --source_csv data/unzipped/nzherald_harvest.csv --train_dir $(dir $@) --test_size 0.15 --log_level INFO

OUTPUT_DIR ?= models/nzherald-gpt2
train: scripts/train.py data/train/train.txt
	$(RUN) python3 $< --train_dir data/train --output_dir $(OUTPUT_DIR) --log_level $(LOG_LEVEL)

tensorboard: DOCKER_ARGS='-p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT)'
tensorbard:
	$(RUN) tensorboard --logdir $(OUTPUT_DIR) host 0.0.0.0 --port $(TENSORBOARD_PORT)

lint:
	$(RUN) black .

clean:
	rm data/unzipped/nzherald_harvest.csv data/train/*

python_shell:
	$(RUN) ipython --no-autoindent

JUPYTER_PASSWORD ?= jupyter
JUPYTER_PORT ?= 8888
.PHONY: jupyter
jupyter: UID=root
jupyter: GID=root
jupyter: DOCKER_ARGS=-u $(UID):$(GID) --rm -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -e NB_USER=$$USER -e NB_UID=$(UID) -e NB_GID=$(GID)
jupyter:
	$(RUN) jupyter lab \
		--allow-root \
		--port $(JUPYTER_PORT) \
		--ip 0.0.0.0 \
		--NotebookApp.password=$(shell $(RUN) \
			python3 -c \
			"from IPython.lib import passwd; print(passwd('$(JUPYTER_PASSWORD)'))")

docker:
	docker build $(DOCKER_ARGS) --tag $(IMAGE):$(GIT_TAG) .
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

docker-push:
	docker push $(IMAGE):$(GIT_TAG)
	docker push $(IMAGE):latest

docker-pull:
	docker pull $(IMAGE):$(GIT_TAG)
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

enter:
	$(RUN) bash

enter-root: UID=root
enter-root: GID=root
enter-root:
	$(RUN) bash
