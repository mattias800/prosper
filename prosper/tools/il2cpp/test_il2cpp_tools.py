#!/usr/bin/env python3
"""
Test suite for IL2CPP tools (resolve.py)

Tests cover:
- Address resolution accuracy
- Edge cases and error handling
- Integration workflow simulation

Siblings, so the split stays clear: test_prx_to_elf.py covers the ELF writer, and
test_symtab_agreement.py covers --emit-symtab against prosper's in-process resolver.
This suite is the unit-level half neither of those has: resolve.py's loader and its
nearest-preceding lookup, driven directly rather than through a CLI.

Run with: python3 test_il2cpp_tools.py -v
"""

import json
import os
import sys
import tempfile
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resolve import load, resolve_one


class TestResolveBasic(unittest.TestCase):
    """Tests for resolve.py address resolution"""

    def setUp(self):
        """Create temporary script.json for testing"""
        self.test_dir = tempfile.mkdtemp(prefix="resolve_test_")
        self.script_path = os.path.join(self.test_dir, "script.json")

        # Create sample script.json with known methods
        methods = [
            {"Address": 0x1000, "Name": "TestClass$$MethodA"},
            {"Address": 0x1100, "Name": "TestClass$$MethodB"},
            {"Address": 0x1200, "Name": "TestClass$$MethodC"},
            {"Address": 0x2000, "Name": "OtherClass$$MethodD"},
            {"Address": 0x2100, "Name": "OtherClass$$MethodE"},
        ]

        with open(self.script_path, 'w') as f:
            json.dump({"ScriptMethod": methods}, f)

        self.addrs, self.keys = load(self.script_path)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_exact_address_match(self):
        """Test resolution of exact method start addresses"""
        result = resolve_one(self.addrs, self.keys, 0x1000)
        self.assertIn("TestClass$$MethodA", result)
        self.assertIn("+0x0", result)

    def test_address_within_method(self):
        """Test resolution of addresses within method body"""
        result = resolve_one(self.addrs, self.keys, 0x1050)
        self.assertIn("TestClass$$MethodA", result)
        self.assertIn("+0x50", result)

    def test_unresolved_address(self):
        """Test that out-of-range addresses return unresolved message"""
        result = resolve_one(self.addrs, self.keys, 0xFFFF)
        self.assertIn("no managed method", result)

    def test_tolerance_boundary(self):
        """Pin BOTH sides of the NEAREST_WINDOW edge, at the edge itself.

        The rule is `off - start < NEAREST_WINDOW`, so the probe has to sit on the last
        method start (0x2100). Measuring from an earlier one never reaches its edge --
        a later start always intervenes and becomes the match -- and an arm a byte
        outside (0x8001) passes whether the comparison is `<` or `<=`, so it cannot see
        an off-by-one. 0x7fff must resolve and exactly 0x8000 must not.
        """
        last = 0x2100
        inside = resolve_one(self.addrs, self.keys, last + 0x7FFF)
        self.assertIn("OtherClass$$MethodE", inside)
        self.assertIn("+0x7fff", inside)

        outside = resolve_one(self.addrs, self.keys, last + 0x8000)
        self.assertIn("no managed method", outside)


class TestResolveEdgeCases(unittest.TestCase):
    """Tests for edge cases in resolve.py"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="resolve_edge_test_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_script_json(self):
        """Test handling of empty method list"""
        script_path = os.path.join(self.test_dir, "empty.json")
        with open(script_path, 'w') as f:
            json.dump({"ScriptMethod": []}, f)

        addrs, keys = load(script_path)
        self.assertEqual(len(addrs), 0)

        result = resolve_one(addrs, keys, 0x1000)
        self.assertIn("no managed method", result)

    def test_missing_address_field(self):
        """Test handling of methods without Address field"""
        script_path = os.path.join(self.test_dir, "no_addr.json")
        methods = [
            {"Name": "MethodA"},  # No Address
            {"Address": 0x2000, "Name": "MethodB"},
        ]
        with open(script_path, 'w') as f:
            json.dump({"ScriptMethod": methods}, f)

        addrs, keys = load(script_path)
        # Should only have one method
        self.assertEqual(len(addrs), 1)

    def test_single_method(self):
        """Test resolution with only one method"""
        script_path = os.path.join(self.test_dir, "single.json")
        with open(script_path, 'w') as f:
            json.dump({"ScriptMethod": [{"Address": 0x1000, "Name": "OnlyMethod"}]}, f)

        addrs, keys = load(script_path)
        result = resolve_one(addrs, keys, 0x1050)
        self.assertIn("OnlyMethod", result)

    def test_large_utf8_tie_keeps_selection_and_reports_all_raw_candidates(self):
        script_path = os.path.join(self.test_dir, "large-tie.json")
        methods = [{'Address': 0x1000, 'Name': f'Tie.Method{i:04d}'}
                   for i in range(2521)]
        methods.extend([{'Address': 0x1000, 'Name': 'é.Type$$Method'},
                        {'Address': 0x1000, 'Name': 'Ω.Type$$Method'},
                        {'Address': 0x11000, 'Name': 'Unique.Next$$Run'}])
        with open(script_path, 'w', encoding='utf-8') as stream:
            json.dump({'ScriptMethod': list(reversed(methods))}, stream)
        addrs, keys = load(script_path)
        self.assertEqual(len(addrs), 2524)
        self.assertEqual(addrs[2522], (0x1000, 'Ω.Type$$Method'))
        for offset in (0, 7, 0x7fff):
            with self.subTest(offset=offset):
                self.assertEqual(resolve_one(addrs, keys, 0x1000 + offset),
                                 f'Ω.Type$$Method  (+0x{offset:x}) (+2522 more at this address)')
        self.assertEqual(resolve_one(addrs, keys, 0x9000),
                         '<il2cpp runtime / no managed method at this offset>')
        self.assertEqual(resolve_one(addrs, keys, 0x11000), 'Unique.Next$$Run  (+0x0)')

    def test_identical_duplicate_rows_still_count_as_two_candidates(self):
        script_path = os.path.join(self.test_dir, "duplicate-tie.json")
        methods = [{'Address': 0x1000, 'Name': 'Duplicate$$Run'},
                   {'Address': 0x1000, 'Name': 'Duplicate$$Run'},
                   {'Address': 0x2000, 'Name': 'Unique$$Run'}]
        with open(script_path, 'w', encoding='utf-8') as stream:
            json.dump({'ScriptMethod': methods}, stream)
        addrs, keys = load(script_path)
        self.assertEqual(keys, [0x1000, 0x1000, 0x2000])
        self.assertEqual(resolve_one(addrs, keys, 0x1001),
                         'Duplicate$$Run  (+0x1) (+1 more at this address)')
        self.assertEqual(resolve_one(addrs, keys, 0x2001), 'Unique$$Run  (+0x1)')


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="integration_test_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_realistic_script_json(self):
        """Test that realistic Il2CppDumper output loads correctly"""
        script_path = os.path.join(self.test_dir, "realistic.json")

        # Simulate realistic Il2CppDumper output structure
        methods = [
            {"Address": 0x0, "Name": "UnityEngine.CoreModule$$Object$$.ctor"},
            {"Address": 0x30, "Name": "UnityEngine.CoreModule$$Object$$ToString"},
            {"Address": 0x80, "Name": "UnityEngine.CoreModule$$Object$$Equals"},
            {"Address": 0xB0, "Name": "UnityEngine.CoreModule$$Object$$GetHashCode"},
            {"Address": 0x100, "Name": "GameManager$$Awake"},
            {"Address": 0x200, "Name": "GameManager$$Start"},
            {"Address": 0x300, "Name": "GameManager$$Update"},
            {"Address": 0x400, "Name": "PlayerController$$Move"},
            {"Address": 0x500, "Name": "PlayerController$$Jump"},
            {"Address": 0x600, "Name": "UIManager$$ShowMenu"},
        ]

        with open(script_path, 'w') as f:
            json.dump({"ScriptMethod": methods}, f)

        addrs, keys = load(script_path)

        # Verify all methods loaded
        self.assertEqual(len(addrs), len(methods))

        # Test resolution at various points
        result1 = resolve_one(addrs, keys, 0x10)  # In .ctor
        self.assertIn("Object$$.ctor", result1)

        result2 = resolve_one(addrs, keys, 0x350)  # In Update
        self.assertIn("GameManager$$Update", result2)

        result3 = resolve_one(addrs, keys, 0x9000)  # Well past all methods (0x600 + 0x8000 = 0x8600)
        self.assertIn("no managed method", result3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
