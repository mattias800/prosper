"""Explicit, narrow source associations for lexical diagnostic reference outputs.

This module validates a maintained source contract; it is not a C++ effects analyzer.
Only literal, uniquely named declarations/definitions, reference aliases and aggregate
bindings are admitted. Source/call identities stay separate from member-name facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

MARKER = re.compile(r'^\s*//\s*PROSPER_DIAG_REF_OUTPUTS\s+(\w+)\s+(\w+)\s*:\s*(.*?)\s*$')
IDENT = re.compile(r'\b[A-Za-z_]\w*\b')
RAW_LITERAL = re.compile(r'R"([^\s()\\]{0,16})\(')
PRIMITIVE_REF = re.compile(r'^\s*(?:bool|int|uint32_t|uint64_t|size_t|double|float)\s*&\s*(\w+)\s*;\s*$')
ALIAS = re.compile(r'^\s*auto\s*&\s*(\w+)\s*=\s*(\w+)\.(\w+)\s*;\s*$')
LAMBDA_INTRO = re.compile(r'\[([^\[\]\n]*)\]\s*(?:\([^{};]*\))?\s*'
                          r'(?:(?:mutable|constexpr|consteval)\s*|noexcept(?:\s*\([^{};]*\))?\s*)*'
                          r'(?:->\s*[^{};]+)?\s*\{')
# Bound recognizable capture/body shapes even when their specifier text is not
# admitted. This deliberately errs toward refusal (for example, array/braced
# syntax can resemble a closure); it is not a full C++ lambda parser.
LAMBDA_BOUNDARY = re.compile(r'\[([^\[\]\n]*)\]\s*[^{};]*?\{')


def bare_identifiers(text: str) -> set[str]:
    result = set()
    for match in IDENT.finditer(text):
        before = match.start()
        while before and text[before-1].isspace():
            before -= 1
        if text[before-1:before] != '.' and text[max(0,before-2):before] not in ('->', '::'):
            result.add(match[0])
    return result


def mask_literals(text: str) -> str:
    """Preserve positions/newlines while hiding ordinary/raw strings and characters."""
    chars = list(text)
    pos = 0
    while pos < len(text):
        if text.startswith('//', pos):
            end = text.find('\n', pos)
            pos = len(text) if end < 0 else end + 1
            continue
        if text.startswith('/*', pos):
            end = text.find('*/', pos + 2)
            if end < 0:
                raise ValueError('unterminated block comment')
            end += 2
            for index in range(pos, end):
                if chars[index] != '\n':
                    chars[index] = ' '
            pos = end
            continue
        raw = RAW_LITERAL.match(text, pos)
        if raw:
            end = text.find(')' + raw[1] + '"', raw.end())
            if end < 0:
                raise ValueError('unterminated raw literal')
            end += len(raw[1]) + 2
        elif text[pos] in '\"\'' and not (
                text[pos] == "'" and pos and pos + 1 < len(text) and
                text[pos-1].isalnum() and text[pos+1].isalnum()):
            quote, end = text[pos], pos + 1
            while end < len(text):
                if text[end] == '\\':
                    end += 2
                elif text[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
            else:
                raise ValueError('unterminated ordinary literal')
        else:
            pos += 1
            continue
        for index in range(pos, end):
            if chars[index] != '\n':
                chars[index] = ' '
        pos = end
    return ''.join(chars)


def matching(text: str, start: int, opening: str, closing: str) -> int:
    assert text[start] == opening
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError('unbalanced ' + opening + closing)


def brace_ancestors(text: str, position: int) -> tuple[int, ...]:
    stack = []
    for index, char in enumerate(text[:position]):
        if char == '{':
            stack.append(index)
        elif char == '}':
            if not stack:
                raise ValueError('unbalanced caller brace scope')
            stack.pop()
    return tuple(stack)


@dataclass
class OutputAssociation:
    marker_path: str
    marker_line: int
    function: str
    parameter: str
    fields: tuple[str, ...]
    context: str = ''
    parameter_index: int = -1
    source_path: str = ''
    source_start: int = 0
    source_end: int = 0
    aliases: dict[str, str] = field(default_factory=dict)
    alias_lines: set[int] = field(default_factory=set)
    binding_lines: set[tuple[str, int, str]] = field(default_factory=set)
    calls: dict[tuple[str, int], dict[str, str]] = field(default_factory=dict)
    writes: dict[str, list[tuple[int, frozenset]]] = field(default_factory=dict)
    value_consumers: dict[str, dict] = field(default_factory=dict)
    calling_closures: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return not self.errors and bool(self.calls)

    def report(self) -> str:
        status = 'admitted' if self.admitted else 'refused'
        reason = '; '.join(self.errors) if self.errors else 'literal reference-output contract'
        return (f'reference outputs {status}: {self.marker_path}:{self.marker_line} '
                f'{self.function}/{self.parameter}, fields={len(self.fields)}, '
                f'calls={len(self.calls)}, value-consumers={len(self.value_consumers)}, '
                f'calling-closures={len(self.calling_closures)}; {reason}')


def _signature(line: str, name: str):
    # Supported literal one-line signatures; overloads and compound declarators refuse.
    match = re.match(r'^\s*([\w:]+)\s+' + re.escape(name) + r'\s*\(([^()]*)\)\s*([;{])\s*$', line)
    if not match:
        return None
    return [part.strip() for part in match[2].split(',')], match[3], match.start(3)


def discover_output_associations(root: Path, files: dict[Path, list[str]], defaulted_local):
    """Return admitted/refused records; absent markers create no inferred associations."""
    records = []
    # Comments are the contract syntax; string-literal marker lookalikes are not declarations.
    raw = {}
    for path in files:
        if path.suffix not in ('.h', '.hpp'):
            continue
        text = path.read_text(encoding='utf-8')
        if 'PROSPER_DIAG_REF_OUTPUTS' in text:
            raw[path] = mask_literals(text).splitlines()
    for path, lines in raw.items():
        for index, line in enumerate(lines):
            match = MARKER.match(line)
            if not match:
                if re.match(r'^\s*//\s*PROSPER_DIAG_REF_OUTPUTS\b', line):
                    record = OutputAssociation(str(path.relative_to(root)), index+1, '', '', ())
                    record.errors.append('malformed reference-output marker')
                    records.append(record)
                continue
            fields = tuple(part.strip() for part in match[3].split(','))
            record = OutputAssociation(str(path.relative_to(root)), index+1,
                                       match[1], match[2], fields)
            records.append(record)
            try:
                if not fields or any(not re.fullmatch(r'[A-Za-z_]\w*', value) for value in fields) or len(set(fields)) != len(fields):
                    raise ValueError('invalid or duplicate selected fields')
                # Exactly the next nonempty line declares the associated private function.
                next_index = next(i for i in range(index+1, len(lines)) if lines[i].strip())
                signature = _signature(lines[next_index], record.function)
                if signature is None or signature[1] != ';':
                    raise ValueError('marker lacks supported following declaration')
                params = signature[0]
                matches = [(i, re.fullmatch(r'(\w+)\s*&\s*' + re.escape(record.parameter), value))
                           for i, value in enumerate(params)]
                matches = [(i, match) for i, match in matches if match]
                if len(matches) != 1:
                    raise ValueError('context must be one literal nonconst reference parameter')
                record.parameter_index, context_match = matches[0]
                record.context = context_match[1]
                _validate(record, root, files, defaulted_local, params)
            except (ValueError, StopIteration) as error:
                record.errors.append(str(error))
    duplicate_names = {r.function for r in records if sum(x.function == r.function for x in records) != 1}
    for record in records:
        if record.function in duplicate_names:
            record.errors.append('duplicate association function')
    return records


def _validate(record, root, files, defaulted_local, declared_params):
    structural = {path: mask_literals('\n'.join(lines)) for path, lines in files.items()}
    context_defs = []
    definitions = []
    signatures = []
    for path, text in structural.items():
        for match in re.finditer(r'\bstruct\s+' + re.escape(record.context) + r'\s*\{', text):
            end = matching(text, match.end()-1, '{', '}')
            context_defs.append(text[match.end():end].splitlines())
        for index, line in enumerate(text.splitlines()):
            signature = _signature(line, record.function)
            if signature is not None:
                signatures.append((path,index+1,signature))
                if signature[1] == '{':
                    start = sum(len(value)+1 for value in text.splitlines()[:index]) + signature[2]
                    end = matching(text,start,'{','}')
                    definitions.append((path,index+1,text.count('\n',0,end)+1,text[start+1:end],signature[0]))
    if len(context_defs) != 1:
        raise ValueError('context declaration is missing or ambiguous')
    primitive = [m[1] for line in context_defs[0] if (m := PRIMITIVE_REF.match(line))]
    if any(primitive.count(name) != 1 for name in record.fields):
        raise ValueError('selected output is not a unique primitive reference field')
    if len(definitions) != 1 or len(signatures) != 2:
        raise ValueError('function declaration/definition is missing, overloaded or ambiguous')
    path, first, last, body, defined_params = definitions[0]
    if defined_params != declared_params:
        raise ValueError('declaration/definition parameter mismatch')
    record.source_path, record.source_start, record.source_end = str(path.relative_to(root)),first,last
    aliases = [m for line in body.splitlines() if (m := ALIAS.match(line)) and m[2] == record.parameter]
    for name in record.fields:
        matches = [m[1] for m in aliases if m[3] == name]
        if len(matches) != 1:
            raise ValueError('output alias missing, rebound or ambiguous: ' + name)
        record.aliases[matches[0]] = name
    record.alias_lines = {first + index for index, line in enumerate(body.splitlines())
                          if (match := ALIAS.match(line)) and match[1] in record.aliases}
    # The context parameter may only initialize exact reference aliases; direct writes/passes refuse.
    without_aliases = '\n'.join(line for line in body.splitlines()
                                if not ((m := ALIAS.match(line)) and m[2] == record.parameter))
    if record.parameter in bare_identifiers(without_aliases):
        raise ValueError('context use bypasses reference aliases')
    _validate_output_uses(body, record.aliases, record, structural, root)
    _validate_calls(record, root, files, structural, signatures, defaulted_local)


def _validate_output_uses(body: str, aliases: dict[str, str], record, structural, root):
    # These are conservative mechanical guards for the reviewed source contract,
    # not a type/effects interpreter. Unknown indirection refuses an association.
    for alias in aliases:
        word = r'\b' + re.escape(alias) + r'\b'
        declaration = re.compile(r'\b(\w+(?:::\w+)*)\s*[&*]*\s+' + re.escape(alias) + r'\b')
        declarations = [m for m in declaration.finditer(body)
                        if m[1] not in ('return', 'case', 'else', 'throw')]
        if len(declarations) != 1 or declarations[0][1] != 'auto':
            raise ValueError('output alias shadowed or redeclared: ' + alias)
        uses = '\n'.join(line for line in body.splitlines() if not ALIAS.match(line))
        if re.search(r'(?<![&\w])&\s*' + word, uses) or re.search(
                r'\bauto\s*[&*]\s*\w+\s*=\s*' + word, uses):
            raise ValueError('output alias address/reference escapes: ' + alias)
        if re.search(word + r'\s*(?:\+\+|--|\*=|/=|%=|\|=|&=|\^=)', body) or re.search(
                r'(?:\+\+|--)\s*' + word, body):
            raise ValueError('unsupported output write operator: ' + alias)
    _validate_closure_uses(body, set(aliases), body_scope=True)
    _refuse_unknown_calls(body, set(aliases), record, structural, root, record.source_path,
                         record.source_start-1)


def _validate_closure_uses(body: str, names: set[str], associated_calls=(), body_scope=False):
    # A selected output may be used through a finite chain of named local [&]
    # closures. Validate every caller transitively: a retained/unknown caller is
    # an escape even if its own body names only another closure, not the output.
    closures = [(match, matching(body,match.end()-1,'{','}'))
                for match in LAMBDA_BOUNDARY.finditer(body)]
    pending = [index for index, (match, end) in enumerate(closures)
               if bare_identifiers(body[match.end():end]) & names
               or set(IDENT.findall(match[1])) & names
               or any(match.start() < call < end for call in associated_calls)]
    checked = set()
    dependencies = {}
    identities = {}
    invisible_uses = []
    while pending:
        index = pending.pop()
        if index in checked:
            continue
        checked.add(index)
        match, end = closures[index]
        supported = LAMBDA_INTRO.match(body, match.start())
        if supported is None or supported.end() != match.end():
            raise ValueError('unsupported output closure specifiers')
        before = body[:match.start()].split('\n')[-1]
        named = re.fullmatch(r'\s*(?:const\s+)?auto\s+(\w+)\s*=\s*', before)
        if named is None or match[1].strip() != '&':
            raise ValueError('output closure is not a supported local reference closure')
        name = named[1]
        declaration_at = match.start()-len(before)+named.start(1)
        scope = brace_ancestors(body, match.start())
        if not body_scope and not any(body[:opening].rstrip().endswith(')') for opening in scope):
            raise ValueError('output closure lacks a recognizable local body')
        identities[index] = {'name': name, 'line': body.count('\n',0,match.start())+1, 'calls': []}
        declarations = list(re.finditer(r'\bauto\s+' + re.escape(name) + r'\s*=', body))
        if len(declarations) != 1:
            raise ValueError('output closure is shadowed or redeclared: ' + name)
        for use in re.finditer(r'\b' + re.escape(name) + r'\b', body):
            if use.start() == declaration_at:
                continue  # Its one local declaration.
            tail = body[use.end():].lstrip()
            if not tail.startswith('(') or match.start() <= use.start() <= end:
                raise ValueError('output closure escapes or recursively calls itself: ' + name)
            identities[index]['calls'].append(body.count('\n',0,use.start())+1)
            if use.start() < declaration_at or brace_ancestors(body,use.start())[:len(scope)] != scope:
                invisible_uses.append(name)
            containers = [other_index for other_index, (other, other_end) in enumerate(closures)
                          if other.start() < use.start() < other_end]
            if containers:
                caller = max(containers, key=lambda other_index: closures[other_index][0].start())
                dependencies.setdefault(index, set()).add(caller)
                pending.append(caller)
    visiting, visited = set(), set()
    def acyclic(index):
        if index in visiting:
            raise ValueError('output closure dependency cycle')
        if index in visited:
            return
        visiting.add(index)
        for caller in dependencies.get(index, ()):
            acyclic(caller)
        visiting.remove(index)
        visited.add(index)
    for index in checked:
        acyclic(index)
    if invisible_uses:
        raise ValueError('output closure call is outside its declared local visibility: ' + invisible_uses[0])
    return [identities[index] for index in sorted(checked)]


def _primitive_value_positions(params: str) -> set[int]:
    primitive = r'(?:bool|int|uint32_t|uint64_t|size_t|double|float)'
    return {index+1 for index, parameter in enumerate(params.split(',')) if re.fullmatch(
        r'\s*(?:const\s+)?' + primitive + r'\s+[A-Za-z_]\w*(?:\s*=\s*[^,]+)?\s*', parameter)}


def _local_value_consumers(text: str, body_scope: bool = False) -> dict[str, dict]:
    candidates = {}
    for match in LAMBDA_INTRO.finditer(text):
        if match[1].strip():
            continue  # Only captureless callables cannot retain caller references.
        params = re.match(r'\[\s*\]\s*\(([^()]*)\)', match[0])
        before = text[:match.start()].split('\n')[-1]
        named = re.fullmatch(r'\s*(?:const\s+)?auto\s+(\w+)\s*=\s*', before)
        if params is None or named is None:
            continue
        name = named[1]
        declared_at = match.start()-len(before)+named.start(1)
        scope = brace_ancestors(text, match.start())
        # A callee slice is already one verified function body. In a complete
        # caller file, require a recognizable enclosing function/control body;
        # a namespace/global callable is outside the local contract.
        if not body_scope and not any(text[:opening].rstrip().endswith(')') for opening in scope):
            continue
        uses = list(re.finditer(r'\b' + re.escape(name) + r'\b', text))
        if any(use.start() != declared_at and
               (use.start() < declared_at or not text[use.end():].lstrip().startswith('(')
                or brace_ancestors(text, use.start())[:len(scope)] != scope) for use in uses):
            continue
        candidate = {'kind': 'local-captureless', 'line': text.count('\n',0,match.start())+1,
                     'positions': sorted(_primitive_value_positions(params[1]))}
        candidates.setdefault(name, []).append(candidate)
    return {name: values[0] for name, values in candidates.items() if len(values) == 1}


def _header_value_consumer(structural, qualified: str, root: Path):
    """Literal single-namespace header signatures; no C++ overload-resolution claim."""
    if '::' not in qualified:
        return None
    namespace, name = qualified.rsplit('::',1)
    candidates = []
    namespace_re = re.compile(r'^[ \t]*(?:(inline)\s+)?namespace\s*([\w:]+)?\s*\{', re.M)
    declaration_re = re.compile(r'^[ \t]*[\w:<>*& ]+\s+(?P<name>' + re.escape(name)
                                + r')\s*\((?P<params>[^()\n]*)\)\s*([;{])\s*$', re.M)
    for path, text in structural.items():
        namespaces = list(namespace_re.finditer(text))
        namespace_openings = {match.end()-1 for match in namespaces}
        for occurrence in re.finditer(r'\b'+re.escape(qualified)+r'\b', text):
            # A recognizable fully qualified declaration/import at file or
            # namespace scope cannot be silently ignored. Ordinary calls inside
            # function/class bodies are not declaration evidence.
            if all(opening in namespace_openings for opening in brace_ancestors(text,occurrence.start())):
                return None
        relevant = [match for match in namespaces if match[2] == namespace]
        if not relevant:
            continue
        if path.suffix not in ('.h', '.hpp'):
            for ns in relevant:
                opening = ns.end()-1
                end = matching(text,opening,'{','}')
                scope = brace_ancestors(text,opening) + (opening,)
                if any(opening < occurrence.start() < end
                       and brace_ancestors(text,occurrence.start()) == scope
                       for occurrence in re.finditer(r'\b'+re.escape(name)+r'\b', text)):
                    return None
            continue
        # Multiple/nested/inline scopes and imported overloads are outside this
        # literal contract. Uncertainty refuses, rather than resolving names globally.
        if len(namespaces) != 1 or relevant[0][1]:
            return None
        ns = relevant[0]
        opening = ns.end()-1
        end = matching(text,opening,'{','}')
        namespace_body = text[opening+1:end]
        if re.search(r'\busing\s+namespace\b|\busing\s+[\w:]+::' + re.escape(name) + r'\s*;', namespace_body):
            return None
        accounted = set()
        for declaration in declaration_re.finditer(text):
            if opening < declaration.start() < end and brace_ancestors(text,declaration.start()) == (opening,):
                accounted.add(declaration.start('name'))
                candidates.append({'kind': 'qualified-header', 'path': str(path.relative_to(root)),
                                   'line': text.count('\n',0,declaration.start())+1,
                                   'positions': sorted(_primitive_value_positions(declaration['params']))})
        # Unsupported/multiline overloads must not vanish from uniqueness. Every
        # recognizable same-name occurrence at direct namespace scope has to be
        # this supported declaration; opaque construction stays outside the contract.
        direct_names = {match.start() for match in re.finditer(r'\b'+re.escape(name)+r'\b', text)
                        if opening < match.start() < end
                        and brace_ancestors(text,match.start()) == (opening,)}
        if direct_names != accounted:
            return None
    return candidates[0] if len(candidates) == 1 else None


def _argument_parts(args: str):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(args):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(args[start:index])
            start = index+1
    parts.append(args[start:])
    return parts


def _refuse_unknown_calls(text: str, names: set[str], record, structural, root, path,
                          line_bias=0, allowed_initializers=()):
    # Casts/control syntax consume values. Other argument-taking calls could write
    # a reference. Ordinary printf report arguments keep the existing lexical
    # rule's semantics; this contract does not interpret formatter effects.
    calls = []
    # Literal braced construction can bind a writable reference too. Distinguish
    # recognizable type/value initializers from control, declaration and lambda
    # bodies; do not infer constructor overloads or extend this to general C++.
    bodies = {match.end()-1 for match in LAMBDA_BOUNDARY.finditer(text)}
    for pattern in (
            r'\bnamespace\s*[\w:]*\s*\{',
            r'\b(?:struct|class|union)\s+\w+(?:\s+final)?\s*(?::[^{};]*)?\{',
            r'\benum(?:\s+class)?\s+\w+\s*(?::[^{};]*)?\{'):
        bodies.update(match.end()-1 for match in re.finditer(pattern,text))
    construction = re.compile(r'\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)'
                              r'(?:\s*<[^{};]*>)?(?:\s+[A-Za-z_]\w*)?\s*\{')
    for brace in construction.finditer(text):
        opening = brace.end()-1
        if opening in bodies or brace[1] in ('else','try','do'):
            continue
        end = matching(text,opening,'{','}')
        if (opening,end) in allowed_initializers:
            continue  # The independently verified exact context object/range.
        if bare_identifiers(text[opening+1:end]) & names:
            raise ValueError('output passed to unknown braced construction')
    # Parenthesized pointers/callables, indexed dispatch and callable-producing
    # expressions are not literal name(...) consumers. Their selected argument
    # still could bind a writable reference; do not silently skip the indirection.
    # Conservatively refuse these recognizable call-shaped suffixes rather than
    # attempting C++ expression/overload interpretation.
    for indirect in re.finditer(r'[)\]}]\s*(?:<[^{};]*>\s*)?\(', text):
        opening = indirect.end()-1
        end = matching(text,opening,'(',')')
        if bare_identifiers(text[opening+1:end]) & names:
            raise ValueError('output passed to unknown indirect argument-taking call')
    local_consumers = _local_value_consumers(text, body_scope=bool(line_bias))
    header_consumers = {}
    for match in re.finditer(r'\b([\w:]+)(?:\s*<[^{};()]*>)?\s*\(', text):
        end = matching(text,match.end()-1,'(',')')
        calls.append((match[1],match.start(),end,text[match.end():end]))
    for name,start,end,args in calls:
        if not (bare_identifiers(args) & names):
            continue
        before = start
        while before and text[before-1].isspace():
            before -= 1
        member_call = text[before-1:before] == '.' or text[max(0,before-2):before] == '->'
        if name in ('if','while','switch','sizeof','alignof','static_cast',
                    'printf','fprintf','std::printf','std::fprintf') and not member_call:
            continue
        consumer = None if member_call else local_consumers.get(name)
        if consumer is None and not member_call:
            if name not in header_consumers:
                header_consumers[name] = _header_value_consumer(structural, name, root)
            consumer = header_consumers[name]
        arguments = _argument_parts(args)
        positions = {index+1 for index, arg in enumerate(arguments)
                     if bare_identifiers(arg) & names}
        direct_values = all(arguments[index-1].strip() in names for index in positions)
        if consumer is not None and direct_values and positions <= set(consumer['positions']):
            identity = dict(consumer)
            if identity['kind'] == 'local-captureless':
                identity['path'] = path
                identity['line'] += line_bias
            previous = record.value_consumers.get(name)
            if previous is not None and any(previous[key] != value for key, value in identity.items()):
                raise ValueError('value consumer has ambiguous source identity: ' + name)
            if previous is None:
                identity['calls'] = []
                record.value_consumers[name] = identity
                previous = identity
            previous['calls'].append({'path': path, 'line': text.count('\n',0,start)+1+line_bias,
                                      'positions': sorted(positions)})
            continue
        raise ValueError('output passed to unknown argument-taking call: ' + name)


def _validate_calls(record, root, files, structural, signatures, defaulted_local):
    signature_locations = {(path,line) for path,line,_ in signatures}
    admitted_objects = set()
    for path,text in structural.items():
        lines = text.splitlines()
        for match in re.finditer(r'\b'+re.escape(record.function)+r'\b',text):
            line = text.count('\n',0,match.start())+1
            if (path,line) in signature_locations:
                continue
            opening = match.end()
            while opening < len(text) and text[opening].isspace():
                opening += 1
            if opening >= len(text) or text[opening] != '(' or (match.start() and text[match.start()-1] in '.:'):
                raise ValueError('function use is not a supported direct call')
            try:
                calling_closures = _validate_closure_uses(text, set(), (match.start(),))
            except ValueError as error:
                raise ValueError('associated call is inside a closure without verified local uses: ' + str(error))
            record.calling_closures.extend(dict(closure, path=str(path.relative_to(root)),
                                               associated_line=line) for closure in calling_closures)
            end = matching(text,opening,'(',')')
            arguments = [value.strip() for value in text[opening+1:end].split(',')]
            if len(arguments) <= record.parameter_index or any(not re.fullmatch(r'\w+',value) for value in arguments):
                raise ValueError('unsupported call argument syntax')
            name = arguments[record.parameter_index]
            aggregate = list(re.finditer(r'\b'+re.escape(record.context)+r'\s+'+re.escape(name)+r'\s*\{',text))
            if len(aggregate) != 1 or aggregate[0].start() >= match.start():
                raise ValueError('context aggregate missing, reused or ambiguous')
            start = aggregate[0].end()-1
            finish = matching(text,start,'{','}')
            bindings = {}
            for item in text[start+1:finish].split(','):
                if not item.strip():
                    continue
                field_match = re.fullmatch(r'\s*\.(\w+)\s*=\s*(\w+)\s*',item)
                if field_match is None or field_match[1] in bindings:
                    raise ValueError('context initializer is not unique literal field-to-local bindings')
                bindings[field_match[1]] = field_match[2]
            if any(field not in bindings for field in record.fields):
                raise ValueError('context lacks selected output binding')
            for field_match in re.finditer(r'\.(\w+)\s*=\s*(\w+)', text[start+1:finish]):
                if field_match[1] in record.fields:
                    binding_line = text.count('\n', 0, start+1+field_match.start())+1
                    record.binding_lines.add((str(path.relative_to(root)), binding_line, field_match[1]))
            object_id = (str(path),aggregate[0].start())
            if object_id in admitted_objects:
                raise ValueError('context object used by multiple calls')
            occurrences = list(re.finditer(r'\b'+re.escape(name)+r'\b',text))
            if len(occurrences) != 2:
                raise ValueError('context object escapes, is passed elsewhere or reused')
            admitted_objects.add(object_id)
            # Output caller locals may be formatted/read; reference aliases, address taking,
            # or unknown calls outside reports are unsupported extra writers.
            for field in record.fields:
                local = bindings[field]
                default_lines = [index for index, value in enumerate(lines)
                                 if defaulted_local(value.strip(), local)]
                if len(default_lines) != 1 or default_lines[0] >= text.count('\n', 0, start):
                    raise ValueError('caller output is not a unique preceding defaulted local')
                default_offset = sum(len(value)+1 for value in lines[:default_lines[0]])
                local_scope = brace_ancestors(text, default_offset)
                call_scope = brace_ancestors(text, match.start())
                if not local_scope or call_scope[:len(local_scope)] != local_scope:
                    raise ValueError('caller defaulted local is outside the call scope')
                declarations = [value for value in lines if re.match(
                    r'^\s*(?:(?:static|const|constexpr|thread_local|inline|unsigned|signed)\s+)*'
                    r'(?!return\b)[\w:<>]+\s*[&*]?\s+' + re.escape(local) + r'\s*(?:=|\{|;)', value)]
                if len(declarations) != 1:
                    raise ValueError('caller output is shadowed or redeclared')
                relevant = [value for value in lines if re.search(r'\b'+re.escape(local)+r'\b',value)]
                if any(re.search(r'(?<![&\w])&\s*\b'+re.escape(local)+r'\b',value) for value in relevant):
                    raise ValueError('caller output address escapes')
                if any(re.search(r'\bauto\s*[&*]\s*\w+\s*=\s*\b'+re.escape(local)+r'\b',value) for value in relevant):
                    raise ValueError('caller output reference escapes')
            caller_names = {bindings[field] for field in record.fields}
            _validate_closure_uses(text, caller_names)
            _refuse_unknown_calls(text, caller_names, record, structural, root,
                                  str(path.relative_to(root)), allowed_initializers=((start,finish),))
            call_key = (str(path.relative_to(root)),line)
            if call_key in record.calls:
                raise ValueError('multiple associated calls on one line')
            record.calls[call_key] = {field:bindings[field] for field in record.fields}
    if not record.calls:
        raise ValueError('associated function has no supported call')
