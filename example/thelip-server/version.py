"""Bumped whenever the server's contract or its models change. /health
carries it; run.py attaches to a running server of the same version and
replaces one of another (an older server left running in its window would
otherwise keep answering after an update)."""
SERVER_VERSION = 7
