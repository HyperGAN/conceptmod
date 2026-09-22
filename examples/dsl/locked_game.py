"""Authored game: one remap phrase, locked_shared critic.

``game(SOURCE).score()`` runs one CPU step. ``verdict == "PASS"`` means
the step used the locked_shared stamp. It does not mean the 2-D write
geometry was re-scored, and it is not a Music or Anima GPU transfer.
"""

SOURCE = """
phrase red=blue
pairing locked_shared
claim locked
"""

EXPECTED_OPS = ["write"]
