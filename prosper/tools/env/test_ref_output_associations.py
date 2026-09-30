"""Discriminating source-contract controls, run by check_diag_gates --selftest."""
from __future__ import annotations

import tempfile
from pathlib import Path

MARKER = '// PROSPER_DIAG_REF_OUTPUTS produce binding: hits\n'
DECL = 'void produce(Context& binding);\n'
HEADER = 'struct Context {\n    uint64_t& hits;\n};\n' + MARKER + DECL
SOURCE = '''
void produce(Context& binding) {
    auto& hits = binding.hits;
    if (getenv("PROSPER_OUTPUT_SOURCE")) {
        hits = 7;
    }
}
'''
CALLER = '''
void report() {
    uint64_t hits = 0;
    Context image_binding{
        .hits = hits,
    };
    produce(image_binding);
    fprintf(stderr, "hits=%llu\\n", hits);
}
'''
ORIGINAL = '''
void report() {
    uint64_t hits = 0;
    if (getenv("PROSPER_OUTPUT_SOURCE")) {
        hits = 7;
    }
    fprintf(stderr, "hits=%llu\\n", hits);
}
'''


def run_tests(verbose: bool = False) -> int:
    # CLI execution uses __main__; import only after scanner definitions exist.
    import check_diag_gates as scanner

    bad, count = 0, 0

    def check(label, condition, detail=''):
        nonlocal bad, count
        count += 1
        if not condition:
            bad += 1
            print(f'  [FAIL] reference-output: {label}: {detail}')
        elif verbose:
            print(f'  [ok]   reference-output: {label}')

    def scan(header=HEADER, source=SOURCE, caller=CALLER, enabled=True, additions=None):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            contents = {'src/output.hpp': header, 'src/output.cpp': source,
                        'src/caller.cpp': caller}
            contents.update(additions or {})
            for name, text in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
            records = []
            files, _preds, findings = scanner.scan_tree(
                root, follow_outputs=enabled, output_associations=records)
            return records, findings, len(files), scanner.inventory(files)

    def splits(findings, subject='hits'):
        return [finding for finding in findings
                if finding.kind == 'SPLIT-LOCAL' and finding.subject == subject]

    def clauses(*names):
        return frozenset(frozenset((name,)) for name in names)

    def expect(label, source=SOURCE, caller=CALLER, wanted=clauses('PROSPER_OUTPUT_SOURCE'),
               subject='hits', header=HEADER, additions=None):
        records, findings, files, inventory = scan(header, source, caller, additions=additions)
        actual = splits(findings, subject)
        check(label + ' exact missing clauses/count',
              len(actual) == int(bool(wanted)) and (not wanted or actual[0].gates == wanted),
              [finding.line() for finding in actual])
        check(label + ' explicit one admitted contract',
              len(records) == 1 and records[0].admitted, [r.report() for r in records])
        return records, findings, files, inventory

    records, findings, files, inventory = expect('same-spelled reference bindings')
    original = scan('', '', ORIGINAL)[1]
    check('source relocation retains original raw key/clauses/multiplicity',
          [(f.key(), f.gates) for f in splits(findings)] ==
          [(f.key(), f.gates) for f in splits(original)])
    check('callee reference declaration is not an ungated measured write',
          len(records[0].writes['hits']) == 1 and
          records[0].writes['hits'][0][1] == clauses('PROSPER_OUTPUT_SOURCE'))
    check('all files and source environment inventory remain visible',
          files == 3 and 'PROSPER_OUTPUT_SOURCE' in inventory)
    disabled = scan(enabled=False)
    check('bridge-disabled positive loses the named finding',
          not splits(disabled[1]) and len(disabled[0]) == 1)
    absent = scan(header=HEADER.replace(MARKER, ''))
    check('marker-absent positive has zero records and natural lost finding',
          not absent[0] and not splits(absent[1]))

    expect('callee extra gate', source=SOURCE.replace(
        'if (getenv("PROSPER_OUTPUT_SOURCE")) {',
        'if (getenv("PROSPER_OUTPUT_SOURCE")) {\n'
        '        if (getenv("PROSPER_OUTPUT_EXTRA")) {').replace(
        '        hits = 7;', '        hits = 7;\n        }'),
        wanted=clauses('PROSPER_OUTPUT_SOURCE', 'PROSPER_OUTPUT_EXTRA'))
    expect('caller and callee gates conjoin', caller=CALLER.replace(
        '    produce(image_binding);',
        '    if (getenv("PROSPER_OUTPUT_CALL")) {\n'
        '        produce(image_binding);\n    }'),
        wanted=clauses('PROSPER_OUTPUT_SOURCE', 'PROSPER_OUTPUT_CALL'))
    expect('caller braceless gate conjoins', caller=CALLER.replace(
        '    produce(image_binding);',
        '    if (getenv("PROSPER_OUTPUT_CALL")) produce(image_binding);'),
        wanted=clauses('PROSPER_OUTPUT_SOURCE', 'PROSPER_OUTPUT_CALL'))
    expect('alternative source clause stays an alternative', source=SOURCE.replace(
        'getenv("PROSPER_OUTPUT_SOURCE")',
        'getenv("PROSPER_OUTPUT_SOURCE") || getenv("PROSPER_OUTPUT_ALTERNATE")'),
        wanted=frozenset((frozenset(('PROSPER_OUTPUT_SOURCE', 'PROSPER_OUTPUT_ALTERNATE')),)))
    expect('printer implies producer', caller=CALLER.replace(
        '    fprintf(stderr, "hits=%llu\\n", hits);',
        '    if (getenv("PROSPER_OUTPUT_SOURCE")) {\n'
        '        fprintf(stderr, "hits=%llu\\n", hits);\n    }'), wanted=frozenset())
    expect('ordinary ungated caller write retains existing retirement', caller=CALLER.replace(
        '    produce(image_binding);', '    produce(image_binding);\n    hits = 11;'),
        wanted=frozenset())
    expect('callee ungated write retains existing retirement', source=SOURCE.replace(
        '    auto& hits = binding.hits;', '    auto& hits = binding.hits;\n    hits = 11;'),
        wanted=frozenset())
    expect('no qualifying write fabricates no producer', source=SOURCE.replace(
        '        hits = 7;', '        hits = 0;'), wanted=frozenset())
    expect('conditional no-write arm retains lexical non-default producer', source=SOURCE.replace(
        '        hits = 7;', '        if (ordinary_condition) {\n            hits = 7;\n        }'))
    expect('caller local name differs from field', caller=CALLER.replace(
        'uint64_t hits = 0;', 'uint64_t measured = 0;').replace(
        '.hits = hits,', '.hits = measured,').replace(
        '"hits=%llu\\n", hits', '"hits=%llu\\n", measured'), subject='measured')
    short_closure = SOURCE.replace(
        '    if (getenv("PROSPER_OUTPUT_SOURCE")) {\n        hits = 7;\n    }',
        '    auto validate = [&] {\n'
        '        if (getenv("PROSPER_OUTPUT_SOURCE")) {\n            hits = 7;\n        }\n'
        '    };\n    validate();')
    # Existing gather_statement consumes a short stored lambda whole. Keep that
    # limitation explicit; a longer source-local closure reaches the lexical walk
    # after its existing 24-line initializer window, as the actual validation body does.
    expect('short closure retains existing unobserved-write limitation',
           source=short_closure, wanted=frozenset())
    expect('long synchronous named validation closure', source=short_closure.replace(
        '    auto validate = [&] {\n',
        '    auto validate = [&] {\n' + '\n' * 25))
    chain = short_closure.replace('    auto validate = [&] {\n',
        '    auto record = [&] { fprintf(stderr, "local=%llu\\n", hits); };\n'
        '    auto validate = [&] {\n' + '\n' * 25).replace(
        '            hits = 7;', '            hits = 7;\n            record();')
    expect('closed synchronous local closure dependency', source=chain)
    expect('same-spelled member is not an output alias use', source=SOURCE.replace(
        '    auto& hits = binding.hits;',
        '    auto& hits = binding.hits;\n    unknown_writer(unrelated.hits);'))

    caller_closure = CALLER.replace('void report() {',
        'void report() {\n    auto dispatch = [&] {\n' + '\n'*25).rstrip()[:-1] + (
        '    };\n    dispatch();\n}\n')
    records, _findings, _files, _inventory = expect('closed named calling closure', caller=caller_closure)
    check('calling closure keeps exact declaration and direct-use identities',
          len(records[0].calling_closures) == 1 and records[0].calling_closures[0]['name'] == 'dispatch'
          and len(records[0].calling_closures[0]['calls']) == 1)
    expect('transitive closed named calling closure', caller=caller_closure.replace(
        '    dispatch();', '    auto outer = [&] { dispatch(); };\n    outer();'))
    expect('conditional closure invocation keeps definition-body lexical semantics', caller=caller_closure.replace(
        '    dispatch();', '    if (getenv("PROSPER_OUTPUT_INVOKE")) dispatch();'))

    other = '''
void unrelated() {
    uint64_t unrelated_hits = 0;
    OtherContext other{
        .hits = unrelated_hits,
    };
    fprintf(stderr, "other=%llu\\n", unrelated_hits);
}
'''
    records, findings, _files, _inventory = expect('unrelated same-field context',
        caller=CALLER + other,
        additions={'src/other.hpp': 'struct OtherContext {\n uint64_t& hits;\n};\n'})
    check('unrelated field receives no global imported writes', not splits(findings, 'unrelated_hits'))
    same_function = CALLER.replace('uint64_t hits = 0;', 'uint64_t measured = 0;').replace(
        '.hits = hits,', '.hits = measured,').replace('"hits=%llu\\n", hits', '"hits=%llu\\n", measured')
    same_function = same_function.replace('    produce(image_binding);',
        '    uint64_t unrelated_hits = 0;\n'
        '    OtherContext other{\n        .hits = unrelated_hits,\n    };\n'
        '    produce(image_binding);\n    fprintf(stderr, "other=%llu\\n", unrelated_hits);')
    records, findings, _files, _inventory = expect('same-scope unrelated context identity',
        caller=same_function, subject='measured',
        additions={'src/other.hpp': 'struct OtherContext {\n uint64_t& hits;\n};\n'})
    check('same-scope unrelated object receives no imported writes', not splits(findings, 'unrelated_hits'))
    second = CALLER.replace('report()', 'second_report()').replace('image_binding', 'second_binding').replace(
        'uint64_t hits = 0;', 'uint64_t second_hits = 0;').replace(
        '.hits = hits,', '.hits = second_hits,').replace(
        '"hits=%llu\\n", hits', '"hits=%llu\\n", second_hits')
    records, findings, _files, _inventory = expect('shared helper distinct objects', caller=CALLER + second)
    check('shared helper imports each exact object/call separately',
          len(records[0].calls) == 2 and len(splits(findings, 'second_hits')) == 1)

    reader_header = '''
namespace observer {
inline void charge(int reason, uint64_t amount) {
}
}
'''
    reader_source = SOURCE.replace('        hits = 7;',
                                   '        observer::charge(1, hits);\n        hits = 7;')
    records, _findings, _files, _inventory = expect('qualified primitive value consumer',
        source=reader_source, additions={'src/reader.hpp': reader_header})
    check('qualified value consumer records exact header/argument identity',
          records[0].value_consumers['observer::charge']['path'] == 'src/reader.hpp'
          and records[0].value_consumers['observer::charge']['positions'] == [1, 2])
    format_caller = CALLER.replace('    fprintf(stderr, "hits=%llu\\n", hits);',
        '    auto format = [](uint64_t value) { return value; };\n'
        '    fprintf(stderr, "hits=%llu\\n", format(hits));')
    records, _findings, _files, _inventory = expect('captureless primitive value formatter', caller=format_caller)
    check('local value consumer records callable source identity',
          records[0].value_consumers['format']['kind'] == 'local-captureless'
          and records[0].value_consumers['format']['path'] == 'src/caller.cpp'
          and records[0].value_consumers['format']['positions'] == [1])
    value_refusals = [
        ('header-reference', reader_header.replace('uint64_t amount', 'uint64_t& amount'), reader_source, CALLER),
        ('header-pointer', reader_header.replace('uint64_t amount', 'uint64_t* amount'), reader_source, CALLER),
        ('header-overload', reader_header.replace('\n}\n}\n',
            '\n}\ninline void charge(int reason, uint64_t& amount);\n}\n'), reader_source, CALLER),
        ('header-multiline-overload', reader_header.replace('\n}\n}\n',
            '\n}\ninline void charge(\n int reason, uint64_t& amount) {\n}\n}\n'), reader_source, CALLER),
        ('header-unsupported-declarator', reader_header.replace('\n}\n}\n',
            '\n}\ninline void charge(int reason, uint64_t& amount) noexcept;\n}\n'), reader_source, CALLER),
        ('namespace-import', reader_header.replace('namespace observer {',
            'namespace observer {\nusing namespace unknown;'), reader_source, CALLER),
        ('namespace-alias', reader_header + '\nnamespace alias = observer;\n',
            reader_source.replace('observer::charge', 'alias::charge'), CALLER),
        ('nested-namespace', reader_header.replace('inline void charge',
            'namespace inner {\n}\ninline void charge'), reader_source, CALLER),
        ('anonymous-namespace', reader_header.replace('inline void charge',
            'namespace {\n}\ninline void charge'), reader_source, CALLER),
        ('unqualified-basename', reader_header, reader_source.replace('observer::charge', 'charge'), CALLER),
        ('qualified-nested-writer', reader_header,
            reader_source.replace('charge(1, hits)', 'charge(1, unknown_writer(hits))'), CALLER),
        ('qualified-unsupported-expression', reader_header,
            reader_source.replace('charge(1, hits)', 'charge(1, hits + foreign_object)'), CALLER),
        ('local-reference', reader_header, SOURCE, format_caller.replace('uint64_t value', 'uint64_t& value')),
        ('local-pointer', reader_header, SOURCE, format_caller.replace('uint64_t value', 'uint64_t* value')),
        ('local-capture', reader_header, SOURCE, format_caller.replace('[](uint64_t', '[&](uint64_t')),
        ('local-alias', reader_header, SOURCE, format_caller.replace(
            '    fprintf(stderr', '    auto copied = format;\n    fprintf(stderr')),
        ('local-address', reader_header, SOURCE, format_caller.replace(
            '    fprintf(stderr', '    auto ptr = &format;\n    fprintf(stderr')),
        ('local-member-shadow', reader_header, SOURCE, format_caller.replace('format(hits)', 'obj.format(hits)')),
        ('global-formatter', reader_header, SOURCE, format_caller.replace(
            '    auto format = [](uint64_t value) { return value; };\n', '').replace(
            'void report()', 'auto format = [](uint64_t value) { return value; };\nvoid report()')),
        ('report-member-shadow', reader_header, SOURCE, CALLER.replace('fprintf(stderr,', 'obj.printf(stderr,')),
    ]
    for label, reader, source, caller in value_refusals:
        records, findings, _files, _inventory = scan(source=source, caller=caller,
                                                    additions={'src/reader.hpp': reader})
        check('refuse value consumer ' + label + ' without a basename escape',
              len(records) == 1 and not records[0].admitted
              and any('unknown argument-taking' in error for error in records[0].errors)
              and not splits(findings), [record.report() for record in records])

    for label, extra in (
            ('cpp-overload', 'namespace observer {\nvoid charge(int reason, uint64_t& amount) {\n}\n}\n'),
            ('cpp-multiline-overload', 'namespace observer {\nvoid charge(\n int reason, uint64_t& amount) {\n}\n}\n'),
            ('cpp-qualified-definition', 'void observer::charge(int reason, uint64_t& amount) {\n}\n'),
            ('cpp-namespace-import', 'namespace observer {\nusing hidden::charge;\n}\n'),
            ('cpp-qualified-import', 'using observer::charge;\n')):
        records, findings, _files, _inventory = scan(source=reader_source,
            additions={'src/reader.hpp': reader_header, 'src/ambiguous.cpp': extra})
        check('refuse recognizable non-header value consumer ' + label,
              len(records) == 1 and not records[0].admitted
              and any('unknown argument-taking' in error for error in records[0].errors)
              and not splits(findings), [record.report() for record in records])
    expect('ordinary nested qualified call is not overload evidence', source=reader_source,
        additions={'src/reader.hpp': reader_header,
                   'src/unrelated.cpp': 'namespace observer {\nvoid other() {\n charge(1, 2);\n}\n}\n'})
    for label, expression in (
            ('parenthesized', '(writer)(hits)'),
            ('pointer', '(*writer)(hits)'),
            ('qualified-pointer', '(*unknown::writer)(hits)'),
            ('member-pointer', '(object.*writer)(hits)'),
            ('pointer-member', '(object->*writer)(hits)'),
            ('indexed', 'writers[index](hits)'),
            ('returned-callable', 'get_writer()(hits)'),
            ('braced-callable', 'Writer{}(hits)'),
            ('template-operator', 'writer.template operator()<uint32_t>(hits)'),
            ('parenthesized-template-operator', '(writer).operator()<uint32_t>(hits)'),
            ('function-type-template-operator', 'writer.template operator()<void(uint64_t&)>(hits)'),
            ('immediate-lambda', '([](uint64_t& value) { value = 1; })(hits)')):
        for side in ('callee', 'caller'):
            source = SOURCE.replace('        hits = 7;', '        '+expression+';\n        hits = 7;') if side=='callee' else SOURCE
            caller = CALLER.replace('    produce(image_binding);', '    produce(image_binding);\n    '+expression+';') if side=='caller' else CALLER
            records, findings, _files, _inventory = scan(source=source,caller=caller)
            check('refuse recognizable indirect writer '+label+' on '+side,
                  len(records)==1 and not records[0].admitted
                  and any('unknown indirect argument-taking' in error for error in records[0].errors)
                  and not splits(findings), [record.report() for record in records])

    for label, expression in (
            ('temporary', 'Writer{hits}'),
            ('named', 'Writer holder{hits}'),
            ('qualified', 'unknown::Writer holder{hits}'),
            ('templated', 'Writer<uint64_t> holder{hits}'),
            ('nested-report', 'fprintf(stderr, "value", Writer{hits})'),
            ('other-context', 'OtherContext other{ .hits = hits }')):
        for side in ('callee', 'caller'):
            source = SOURCE.replace('        hits = 7;', '        '+expression+';\n        hits = 7;') if side=='callee' else SOURCE
            caller = CALLER.replace('    produce(image_binding);', '    produce(image_binding);\n    '+expression+';') if side=='caller' else CALLER
            records, findings, _files, _inventory = scan(source=source,caller=caller)
            check('refuse recognizable braced construction '+label+' on '+side,
                  len(records)==1 and not records[0].admitted
                  and any('unknown braced construction' in error for error in records[0].errors)
                  and not splits(findings), [record.report() for record in records])
    expect('unrelated braced member is not a selected alias', source=SOURCE.replace(
        '        hits = 7;', '        Writer holder{unrelated.hits};\n        hits = 7;'))
    expect('ordinary braced control body is not a construction', source=SOURCE.replace(
        '        hits = 7;', '        if (ordinary_condition) {\n            hits = 7;\n'
        '        } else {\n            hits = 8;\n        }'))

    refusal_cases = [
        ('malformed-marker', HEADER.replace(MARKER, '// PROSPER_DIAG_REF_OUTPUTS broken\n'), SOURCE, CALLER,
         'malformed'),
        ('duplicate-fields', HEADER.replace('binding: hits', 'binding: hits, hits'), SOURCE, CALLER,
         'duplicate selected'),
        ('missing-field', HEADER.replace('binding: hits', 'binding: missing'), SOURCE, CALLER,
         'primitive reference'),
        ('owning-field', HEADER.replace('uint64_t& hits', 'uint64_t hits'), SOURCE, CALLER,
         'primitive reference'),
        ('const-field', HEADER.replace('uint64_t& hits', 'const uint64_t& hits'), SOURCE, CALLER,
         'primitive reference'),
        ('stale-function', HEADER.replace(DECL, 'void gone(Context& binding);\n'), SOURCE, CALLER,
         'following declaration'),
        ('context-mismatch', HEADER, SOURCE.replace('Context& binding', 'Other& binding'), CALLER,
         'parameter mismatch'),
        ('alias-missing', HEADER, SOURCE.replace('auto& hits = binding.hits;', 'auto& hits = other.hits;'), CALLER,
         'missing, rebound or ambiguous'),
        ('alias-duplicate', HEADER, SOURCE.replace('auto& hits = binding.hits;',
         'auto& hits = binding.hits;\n    auto& another = binding.hits;'), CALLER, 'ambiguous'),
        ('alias-shadow', HEADER, SOURCE.replace('        hits = 7;',
         '        uint64_t hits = 7;'), CALLER, 'shadowed or redeclared'),
        ('alias-rebind', HEADER, SOURCE.replace('        hits = 7;',
         '        auto& rebound = hits;'), CALLER, 'reference escapes'),
        ('direct-field-write', HEADER, SOURCE.replace('        hits = 7;',
         '        binding.hits = 7;'), CALLER, 'bypasses reference aliases'),
        ('output-address', HEADER, SOURCE.replace('        hits = 7;',
         '        unknown_writer(&hits);'), CALLER, 'address/reference escapes'),
        ('unknown-callee-writer', HEADER, SOURCE.replace('        hits = 7;',
         '        unknown_writer(hits);'), CALLER, 'unknown argument-taking'),
        ('unsupported-write', HEADER, SOURCE.replace('        hits = 7;',
         '        ++hits;'), CALLER, 'unsupported output write'),
        ('retained-output-closure', HEADER, SOURCE.replace('        hits = 7;',
         '        cleanup.push_back([&] { hits = 7; });'), CALLER, 'local reference closure'),
        ('returned-local-closure', HEADER, SOURCE.replace('        hits = 7;',
         '        auto validate = [&] { hits = 7; };\n        return validate;'), CALLER, 'escapes'),
        ('aliased-local-closure', HEADER, SOURCE.replace('        hits = 7;',
         '        auto validate = [&] { hits = 7; };\n        auto copied = validate;'), CALLER, 'escapes'),
        ('addressed-local-closure', HEADER, SOURCE.replace('        hits = 7;',
         '        auto validate = [&] { hits = 7; };\n        auto ptr = &validate;'), CALLER, 'escapes'),
        ('transitive-output-closure', HEADER, SOURCE.replace('        hits = 7;',
         '        auto validate = [&] { hits = 7; };\n'
         '        cleanup.push_back([&] { validate(); });'), CALLER, 'local reference closure'),
        ('output-closure-cycle', HEADER, SOURCE.replace('        hits = 7;',
         '        auto left = [&] { hits = 7; right(); };\n'
         '        auto right = [&] { left(); };\n        left();'), CALLER, 'dependency cycle'),
        ('recursive-output-closure', HEADER, SOURCE.replace('        hits = 7;',
         '        auto validate = [&] { hits = 7; validate(); };\n        validate();'), CALLER, 'recursively'),
        ('unknown-caller-writer', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    produce(image_binding);\n    unknown_writer(hits);'), 'unknown argument-taking'),
        ('writer-inside-report', HEADER, SOURCE, CALLER.replace(
         '"hits=%llu\\n", hits', '"hits=%llu\\n", unknown_writer(hits)'), 'unknown argument-taking'),
        ('caller-address', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    produce(image_binding);\n    unknown_writer(&hits);'), 'caller output address'),
        ('caller-reference', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    auto& rebound = hits;\n    produce(image_binding);'), 'caller output reference'),
        ('context-escape', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    produce(image_binding);\n    unknown_writer(image_binding);'), 'context object escapes'),
        ('caller-retained-closure', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    cleanup.push_back([&] { produce(image_binding); });'), 'inside a closure'),
        ('caller-mutable-closure', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    cleanup.push_back([&]() mutable { produce(image_binding); });'), 'inside a closure'),
        ('caller-return-closure', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    cleanup.push_back([&]() -> void { produce(image_binding); });'), 'inside a closure'),
        ('caller-noexcept-closure', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    cleanup.push_back([&]() noexcept { produce(image_binding); });'), 'inside a closure'),
        ('caller-noexcept-expression', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    cleanup.push_back([&]() noexcept(true) { produce(image_binding); });'), 'inside a closure'),
        ('caller-unsupported-specifier', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    cleanup.push_back([&]() [[unknown]] { produce(image_binding); });'), 'inside a closure'),
        ('caller-retained-output', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    produce(image_binding);\n    cleanup.push_back([&] { hits = 9; });'), 'local reference closure'),
        ('named-calling-closure-store', HEADER, SOURCE, caller_closure.replace(
            '    dispatch();', '    cleanup.push_back(dispatch);'), 'inside a closure'),
        ('named-calling-closure-pass', HEADER, SOURCE, caller_closure.replace(
            '    dispatch();', '    retain(dispatch);'), 'inside a closure'),
        ('named-calling-closure-return', HEADER, SOURCE, caller_closure.replace(
            '    dispatch();', '    return dispatch;'), 'inside a closure'),
        ('named-calling-closure-alias', HEADER, SOURCE, caller_closure.replace(
            '    dispatch();', '    auto copy = dispatch;\n    copy();'), 'inside a closure'),
        ('named-calling-closure-address', HEADER, SOURCE, caller_closure.replace(
            '    dispatch();', '    auto ptr = &dispatch;'), 'inside a closure'),
        ('named-calling-closure-value-capture', HEADER, SOURCE,
            caller_closure.replace('dispatch = [&]', 'dispatch = [=]'), 'inside a closure'),
        ('named-calling-closure-transitive-retain', HEADER, SOURCE, caller_closure.replace(
            '    dispatch();', '    auto outer = [&] { dispatch(); };\n    retain(outer);'), 'inside a closure'),
        ('named-calling-closure-out-of-scope', HEADER, SOURCE, caller_closure.replace(
            '    auto dispatch =', '    {\n    auto dispatch =').replace(
            '    dispatch();', '    }\n    dispatch();'), 'inside a closure'),
        ('named-calling-closure-before-declaration', HEADER, SOURCE, caller_closure.replace(
            '    auto dispatch =', '    dispatch();\n    auto dispatch ='), 'inside a closure'),
        ('named-calling-closure-retained-mutable', HEADER, SOURCE, caller_closure.replace(
            'dispatch = [&]', 'dispatch = [&]() mutable').replace(
            '    dispatch();', '    retain(dispatch);'), 'inside a closure'),
        ('named-calling-closure-retained-noexcept', HEADER, SOURCE, caller_closure.replace(
            'dispatch = [&]', 'dispatch = [&]() noexcept').replace(
            '    dispatch();', '    retain(dispatch);'), 'inside a closure'),
        ('named-calling-closure-retained-trailing-return', HEADER, SOURCE, caller_closure.replace(
            'dispatch = [&]', 'dispatch = [&]() -> void').replace(
            '    dispatch();', '    retain(dispatch);'), 'inside a closure'),
        ('same-object-two-calls', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    produce(image_binding);\n    produce(image_binding);'), 'context object escapes'),
        ('unsupported-argument', HEADER, SOURCE, CALLER.replace('produce(image_binding)',
         'produce(*image_binding)'), 'call argument syntax'),
        ('unsupported-binding', HEADER, SOURCE, CALLER.replace('.hits = hits,',
         '.hits = alias(hits),'), 'field-to-local bindings'),
        ('nondefault-caller', HEADER, SOURCE, CALLER.replace('hits = 0;', 'hits = 42;'),
         'preceding defaulted local'),
        ('caller-local-shadow', HEADER, SOURCE, CALLER.replace('    produce(image_binding);',
         '    {\n        uint64_t hits = 42;\n        produce(image_binding);\n    }'), 'shadowed or redeclared'),
        ('caller-local-out-of-scope', HEADER, SOURCE, CALLER.replace('    uint64_t hits = 0;',
         '    {\n        uint64_t hits = 0;\n    }'), 'outside the call scope'),
    ]
    for label, header, source, caller, reason in refusal_cases:
        records, findings, files, inventory = scan(header, source, caller)
        check('refuse ' + label + ' visibly without importing a result',
              len(records) == 1 and not records[0].admitted
              and any(reason in error for error in records[0].errors)
              and not splits(findings), [record.report() for record in records])
        check('refuse ' + label + ' retains files and inventory',
              files == 3 and 'PROSPER_OUTPUT_SOURCE' in inventory)

    ordinary = '''
void standalone_report() {
    unsigned standalone = 0;
    if (getenv("PROSPER_OUTPUT_ORDINARY")) {
        standalone = 9;
    }
    fprintf(stderr, "ordinary=%u\\n", standalone);
}
'''
    for label, refused_header in (
            ('malformed', HEADER.replace(MARKER, '// PROSPER_DIAG_REF_OUTPUTS broken\n')),
            ('owning', HEADER.replace('uint64_t& hits', 'uint64_t hits'))):
        records, findings, _files, inventory = scan(header=refused_header, caller=CALLER + ordinary)
        actual = splits(findings, 'standalone')
        check('refused ' + label + ' preserves ordinary independent local diagnosis',
              len(records) == 1 and not records[0].admitted and len(actual) == 1
              and actual[0].gates == clauses('PROSPER_OUTPUT_ORDINARY')
              and 'PROSPER_OUTPUT_ORDINARY' in inventory, [f.line() for f in actual])

    for label, suffix in (
            ('duplicate-marker', '\n' + MARKER + DECL),
            ('duplicate-definition', '\n' + SOURCE)):
        header, source = (HEADER + suffix, SOURCE) if label == 'duplicate-marker' else (HEADER, SOURCE + suffix)
        records, _findings, _files, _inventory = scan(header, source)
        check('refuse ' + label, records and all(not record.admitted for record in records),
              [record.report() for record in records])
    for label, hidden in (
            ('block-comment', '/*\n' + MARKER + DECL + '*/\n'),
            ('raw-string', 'const char* payload = R"tag(\n' + MARKER + DECL + ')tag";\n'),
            ('ordinary-string', 'const char* payload = "// PROSPER_DIAG_REF_OUTPUTS produce binding: hits";\n')):
        records, _findings, _files, _inventory = scan(HEADER.replace(MARKER, '') + hidden)
        check('hidden ' + label + ' cannot declare association', not records)

    if not bad:
        print(f'  [ok]   reference-output associations: {count} controls; exact call/object, '
              'gates, bridge-off, no-write, visibility and refusal arms')
    return bad


if __name__ == '__main__':
    raise SystemExit(run_tests(True))
