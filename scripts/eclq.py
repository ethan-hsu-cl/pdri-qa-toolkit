#!/usr/bin/env python3
"""``ecl_query`` with this repo's token resolution applied first.

The vendored ``ecl_query.py`` reads ``$ECL_TOKEN`` or ``~/.ecl_token`` only —
it knows nothing about the repo-local ``.ecl_token``, so running it directly
reports "No token" on a machine that is otherwise fully configured. This
resolves the token the way the pipeline does, then hands over unchanged.

    ./scripts/eclq.py tr TR260910-067
    ./scripts/eclq.py bugs --product "PowerDirector Mobile for iOS" --limit 5
    ./scripts/eclq.py bug DRI261883-0064

Same output, same arguments, same confidentiality caveat as ecl_query: query
tokens have no permission filtering, so treat what comes back accordingly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecl_direct import token_status  # noqa: E402
import ecl_query  # noqa: E402

if __name__ == "__main__":
    status = token_status()
    for warning in status["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not status["available"]:
        print(f"\nERROR: {status['reason']}", file=sys.stderr)
        sys.exit(2)
    sys.exit(ecl_query.main())
