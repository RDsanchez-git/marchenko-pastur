# Experimental Domain (I+D)

This directory contains exploratory, non-deterministic, and non-production code.

🚨 **STRICT RULES:**
1. **No API Definition:** Nothing here is part of the public or internal API.
2. **No Upstream Imports:** `src/` and `tests/` MUST NEVER import from this directory.
3. **No Guarantees:** Results and code stability are not guaranteed.

**Workflow:**
Prototype ideas here -> Validate empirically -> Refactor -> Move to `src/` -> Write `tests/`.