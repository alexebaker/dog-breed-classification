#!/bin/bash

virtualenv .venv

.venv/bin/pip install numpy
.venv/bin/pip install -r requirements.txt
