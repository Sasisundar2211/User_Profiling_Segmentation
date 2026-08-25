"""Vercel Serverless Function entry point for Flask application."""

from __future__ import annotations

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Vercel WSGI Handler
app = app
