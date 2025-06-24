#!/bin/bash

pip install -r requirements.txt
gunicorn main:app --bind=0.0.0.0 --timeout 600
