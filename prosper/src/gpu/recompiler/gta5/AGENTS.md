# `recompiler/gta5` — title-specific compute contracts

Evidence-driven lowerings for *Grand Theft Auto V* (`PPSA04263`) compute programs whose behaviour was
established from live captures rather than from a published contract.

Kept apart from the translator on purpose: these churn on a different cadence, they are justified by
one title's measurements, and mixing them into `rdna2_to_spirv` would make it impossible to see which
rules are general and which are a specific game's.

**Adding something here is an admission that the general path could not express it.** Prefer
generalising into the translator when the evidence supports it; when it does not, land it here with
the measurement that justifies it and a `CONFIDENCE:` marker.
