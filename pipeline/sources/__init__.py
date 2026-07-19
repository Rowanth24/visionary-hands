"""Dataset source adapters for the multi-source pipeline (plan Task D / Gates G1-G2).

Each source module normalizes its native metadata to the entry shape the
existing signer-disjoint splitter (eval/prepare_cross_source_finetune_data.py)
already consumes:
    {"label": str, "video_id": str, "filepath": str, "signer_id": str, "source": str}

Sources that cannot be downloaded by script (license gates, bulk-only archives)
raise NeedsManualStep whose message IS the team instruction sheet.
"""


class NeedsManualStep(RuntimeError):
    """Raised when acquiring the data requires a human step (license/bulk download)."""
