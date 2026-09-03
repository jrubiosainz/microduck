#!/usr/bin/env python3
"""The commands the duck can be given: the vocabulary, and nothing else.

A three-line module on purpose.  Both the template table and the classifier need
these names, and neither should have to import the other to get them - a cycle
there is what forces one of two equal peers to become a subordinate.

THE COMMAND IS NOT THE GESTURE.  A gesture is read in the PERSON's frame; the
command is what the DUCK must do.  The instructor faces the duck, so her raised
LEFT arm points to the duck's right.  That mapping lives in the template table,
which is the only place the two frames meet.
"""

from __future__ import annotations

# -- the commands the duck can be given -------------------------------------
CMD_COME = "COME"
CMD_STOP = "STOP"
CMD_TURN_LEFT = "TURN_LEFT"
CMD_TURN_RIGHT = "TURN_RIGHT"
CMD_BACK_UP = "BACK_UP"
CMD_WAVE = "WAVE"

COMMANDS: tuple[str, ...] = (
    CMD_COME, CMD_STOP, CMD_TURN_LEFT, CMD_TURN_RIGHT, CMD_BACK_UP, CMD_WAVE)

