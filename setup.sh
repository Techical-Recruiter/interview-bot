#!/bin/bash
# Force pip upgrade
pip install --upgrade pip
# Install dependencies with binary preference
pip install --no-cache-dir --prefer-binary -r requirements.txt
