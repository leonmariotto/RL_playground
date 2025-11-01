import sys


def print_progress_bar(start, current, end, length=40):
    """
    Print a simple progress bar in the terminal.

    Args:
        start (int): starting index (usually 0)
        current (int): current index
        end (int): last index
        length (int): length of the bar in characters
    """
    # Clamp progress to [0, 1]
    progress = (current - start) / (end - start)
    progress = max(0, min(1, progress))

    filled = int(length * progress)
    bar = "█" * filled + "-" * (length - filled)
    percent = int(progress * 100)

    # '\r' returns cursor to start of line so we overwrite it
    sys.stdout.write(f"\r[{bar}] {percent:3d}%")
    sys.stdout.flush()

    # Print newline when finished
    if current >= end:
        sys.stdout.write("\n")
