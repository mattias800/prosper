# Declared reference outputs for the lexical diagnostic gate

`check_diag_gates.py` detects a defaulted scalar that is written under one environment
requirement and printed without that requirement. Moving its producer into a function
can hide that association from the scanner even when the C++ behavior is unchanged.
Removing its baseline classification would conceal the unchanged coupling.

An explicit source contract can preserve this one association. Immediately before a
private function's one-line declaration in a collected `.h` or `.hpp`, write:

```cpp
// PROSPER_DIAG_REF_OUTPUTS produce binding: hits
void produce(Context& binding);
```

This is a maintained assertion about selected reference outputs. It is not general C++
effect analysis, complete reachability analysis, or a global member-name propagation
rule. It adds no runtime code. A missing marker imports nothing. A present unsupported,
stale or ambiguous marker prints a refusal record and makes the gate fail.

The supported shape is deliberately narrow:

- There is exactly one literal context `struct`, one function declaration and one
  definition with matching parameter text. The marked parameter is `Context& binding`.
- Each selected field is a unique primitive nonconst reference: `bool`, `int`,
  `uint32_t`, `uint64_t`, `size_t`, `double` or `float`.
- The callee binds each output exactly once as `auto& alias = binding.field;`.
  The context parameter only binds literal reference aliases; direct field access,
  passing it elsewhere, alias shadowing, address escape and unknown writers refuse.
- Each call passes literal identifier arguments and one uniquely named literal
  `Context object{ .field = local, ... };`. The object has exactly its declaration
  and this one call use. Distinct objects can call a shared helper; reusing an object
  or putting multiple associated calls on one source line refuses.
- Each caller output is one uniquely declared preceding defaulted local in an
  enclosing brace scope. Its actual visibility is also checked against the scanner's
  declaration stack before any effects are imported. Unknown argument-taking uses,
  reference/address escape and shadowing refuse.
- Selected outputs may be formatted directly by `printf`/`fprintf` (including their
  `std::` spellings) or read through control syntax and `static_cast`. An unknown
  nested call inside a report still refuses; it could write a reference.
- A literal output identifier may be passed to a proven primitive-by-value argument
  of one exact qualified header function, or one exact local captureless lambda.
  The header shape requires a unique declaration/definition at direct scope in a
  single literal combined namespace, without imported, nested, anonymous or inline
  namespace ambiguity. Every recognizable same-name direct-scope occurrence must
  map to the supported signature; multiline/unsupported overloads refuse. The
  same recognizable namespace declarations and qualified definitions/imports in
  collected translation units also revoke the sole header identity; ordinary calls
  inside other function bodies do not count as overload evidence. The
  local lambda requires typed primitive value parameters and only direct, visible
  name uses after its declaration. Ref/pointer arguments, overloads, aliases, unknown
  expressions and same-named member calls refuse. Accepted records retain the
  declaration's header/local identity, argument positions and each consuming call.
  This is a narrow literal type contract, not general overload resolution.
  Parenthesized pointer/callable, member-pointer, indexed or callable-producing
  invocation shapes containing an output also refuse; they cannot silently bypass
  the direct-name guard. Closing `)`, `]` or `}` followed by such an argument list
  is conservatively outside the contract, including an intervening template suffix
  or syntax that only resembles a call.
  Recognizable literal type/value braced constructions containing an output also
  refuse without inferring constructor effects. Control, namespace/type-definition
  and lambda bodies are distinguished from those initializer spans. Only the exact
  independently validated associated context object's initializer range is allowed;
  a same-named field/type in another initializer gains no exception. Other C++
  declaration/initializer grammar remains part of the maintained source assertion.
- A named local `[&]` closure that uses an output may remain only if every use of its
  name is a direct nonrecursive call in the source function or another admitted local
  closure. Every such caller is checked transitively; the dependency graph must be
  acyclic, with each local call after its declaration and inside its brace visibility.
  Passing, aliasing, returning, storing or taking its address refuses. The same
  exact closure/use graph may contain an associated caller invocation; its selected
  defaults, object and call must still satisfy the independent local identity and
  visibility checks. Anonymous or escaping calling closures and other capture
  shapes refuse. Accepted records retain each calling closure's declaration and
  direct use lines. Calling a closure conditionally does not add execution-path
  analysis to the existing lexical definition-body treatment.
  Call containment also checks recognizable capture/body spans with unsupported
  specifier text; it cannot quietly admit the `mutable`, trailing-return or `noexcept`
  forms as ordinary calls. This conservative check can refuse array/braced syntax
  that resembles a closure. Arbitrary C++ lambda grammar remains outside the contract.

Only the selected alias's non-default writes observed by the existing `FileScanner`
are imported. Each write keeps its callee clauses, conjoined with the exact caller
call's clauses. An alternative within a clause remains an alternative. No qualifying
write means no invented producer. Ordinary caller writes still participate in the
existing intersection of write requirements.

Reference alias declarations and the admitted object's selected designated bindings
initialize references, not measurements. Their exact lines are excluded from scalar
write bookkeeping only for a fully admitted association. A refused contract cannot
suppress ordinary diagnostic bookkeeping. Source paths, function ranges, context
declaration offsets and call lines identify each association; another object's
same-named field receives no effects.

The guards reject supported recognizable misuse. They do not evaluate arbitrary
macros, overloaded operators or C++ control flow. The existing scanner's lexical
treatment of local lambda bodies, default resets and braceless source writes remains
unchanged. In particular, a short stored-lambda initializer is consumed whole, while
the existing 24-line initializer window can expose later statements in a longer body;
this bridge preserves that lexical limitation. The source declaration and its lifetime/effect obligations still need
human review; accepted records are not a general compiler proof.

`test_ref_output_associations.py` runs inside both `--selftest` and full scans. Its
positive depends on the bridge: disabling the bridge or removing the marker loses
the named finding. Callee and caller gate changes alter its exact raw clauses; no-write,
ordinary-write and printer-gate controls constrain false positives. Refusal and
visibility controls cover escapes, closures, shadowing, signatures, repeated objects,
hidden marker strings and local/member names that must stay unrelated.
