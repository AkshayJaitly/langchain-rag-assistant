"""Seed the index with the bundled demo documents.

The hosted disk is ephemeral, so every redeploy leaves a visitor looking at an
empty app with nothing to ask about. These three short fictional documents --
a service agreement, a remote-work policy, and a quarterly metrics review --
give the demo something to answer and exercise the parts of extraction worth
showing: tables, headings, and figures spread across pages.

Seeding is skipped as soon as anything has been ingested, so a real upload is
never competing with the samples.
"""
from __future__ import annotations

import logging
import os

from app.config import get_settings
from app.rag.ingest import ingest_file
from app.rag.vectorstore import read_manifest

logger = logging.getLogger("rag")

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")


def sample_files() -> list[str]:
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return [
        os.path.join(SAMPLES_DIR, name)
        for name in sorted(os.listdir(SAMPLES_DIR))
        if name.lower().endswith(".pdf")
    ]


def seed_samples() -> int:
    """Ingest the bundled samples if the index is empty. Returns files added."""
    if read_manifest():
        return 0

    added = 0
    for path in sample_files():
        name = os.path.basename(path)
        try:
            # Samples belong to the public tenant, so every visitor can read
            # them while their own uploads stay private (spec 004 AC-1).
            ingest_file(path, name, get_settings().public_tenant)
            added += 1
        except Exception:  # noqa: BLE001 - a bad sample must not block startup
            logger.exception("Could not seed sample document %s", name)
    if added:
        logger.info("Seeded %d sample document(s)", added)
    return added
