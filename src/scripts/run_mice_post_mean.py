FINAL_SCOPE_MESSAGE = (
    "mice_post_mean is not part of the final thesis benchmark scope. "
    "Use main.py or src/scripts/run_classical_manifest.py for the supported "
    "classical benchmark methods."
)


def run_mice_post_mean() -> None:
    """Run mice post mean."""
    raise SystemExit(FINAL_SCOPE_MESSAGE)


if __name__ == "__main__":
    run_mice_post_mean()
