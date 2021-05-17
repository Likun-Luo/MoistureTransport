#!/bin/bash

# Reformat every python file in the root directory
yapf -i -vv ./*.py
# Reformat every python file in the src directory and its' child directories
yapf -i -r -vv src/.