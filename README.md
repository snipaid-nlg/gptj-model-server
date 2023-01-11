
# Running our finetuned model gpt-j-title-teaser-10k

This repo gives a basic framework for serving ML models in production using simple HTTP servers.

## Quickstart:

The repo is already set up to run our finetuned gpt-j model gptj-title-teaser-10k for title and teaser generation.
1. Run `pip3 install -r requirements.txt` to download dependencies.
2. Run `python3 server.py` to start the server.
3. Run `python3 test.py` in a different terminal session to test against it.

*Note: Model should be run on a GPU!*

## Overview:

1. `app.py` contains the code to load and run your model for inference.
2. You can run a simple test with `test.py`!

if deploying using Docker:

3. `download.py` is a script to download our finetuned model weights at build time.

## Production:

This repo provides you with a functioning http server for our finetuned gptj-title-teaser-10k model. You can use it as is, or package it up with our provided `Dockerfile` and deploy it to your favorite container hosting provider!

## How to deploy:

*Coming soon: Instructions on deployment with example provider...*
