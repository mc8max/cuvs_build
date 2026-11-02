"""Coverage selection must retain all supported declarations and explain omissions."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_test_plan", ROOT / "scripts/test_plan.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


class NativeTestPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cuvs-test-plan-")
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)
        for folder in ("cpp/tests", "c/tests", "c/include/cuvs/core"):
            (self.source / folder).mkdir(parents=True)
        (self.source / "cpp/tests/CMakeLists.txt").write_text('''
# ConfigureTest(NAME COMMENTED_TEST PATH missing.cu)
function(ConfigureTest)
endfunction()
ConfigureTest(NAME ORDINARY_TEST PATH ordinary.cu GPUS 1)
ConfigureTest(NAME NEW_UPSTREAM_TEST PATH ordinary.cu GPUS 1)
ConfigureTest(NAME CUTILE_SMOKE_TEST PATH ordinary.cu)
if(BUILD_MG_ALGOS)
  ConfigureTest(NAME NEIGHBORS_MG_TEST PATH ordinary.cu)
endif()
ConfigureTest(NAME CLUSTER_KMEANS_MNMG_TEST PATH ordinary.cu)
''')
        (self.source / "cpp/tests/ordinary.cu").write_text("TEST(Suite, Example) {}\n")
        (self.source / "c/tests/CMakeLists.txt").write_text('''
ConfigureTest(NAME cuvs_c_headers PATH command.c)
ConfigureTest(NAME cuvs_c_neighbors_test PATH compile_only.c)
ConfigureTest(NAME cuvs_c_resource_pool_test PATH test.cpp)
ConfigureTest(NAME MG_C_TEST PATH test.cpp)
''')
        (self.source / "c/tests/command.c").write_text("int main(void) { return 0; }\n")
        (self.source / "c/tests/compile_only.c").write_text("int main(void) { return 0; }\n")
        (self.source / "c/tests/test.cpp").write_text("TEST(Resources, Example) {}\n")
        (self.source / "c/include/cuvs/core/a.h").write_text("#pragma once\n")
        (self.source / "c/include/cuvs/core/all.h").write_text("#include <cuvs/core/a.h>\n")

    def plan(self, **kwargs):
        return planner.create_plan(self.source, "12.9", {"gpus": [{"name": "MIG GPU"}]}, **kwargs)

    def test_new_upstream_tests_are_included_without_an_allowlist(self):
        names = {entry["name"] for entry in self.plan()["tests"]}
        self.assertIn("NEW_UPSTREAM_TEST", names)
        self.assertNotIn("COMMENTED_TEST", names)

    def test_plain_mains_are_commands_and_compile_only_never_executes(self):
        plan = self.plan()
        by_name = {entry["name"]: entry for entry in plan["tests"]}
        self.assertEqual(by_name["cuvs_c_headers"]["kind"], "command")
        self.assertEqual(by_name["cuvs_c_resource_pool_test"]["kind"], "gtest")
        self.assertEqual(by_name["cuvs_c_resource_pool_test"]["gtest_filter"], "*")
        self.assertNotIn("cuvs_c_neighbors_test", by_name)
        self.assertIn("cuvs_c_neighbors_test", plan["build_targets"])
        self.assertEqual(plan["build_only"][0]["name"], "cuvs_c_neighbors_test")

    def test_all_declarations_have_a_reported_outcome(self):
        plan = self.plan()
        discovered = {entry["name"] for entry in planner.discover_tests(self.source)}
        reported = [entry["name"] for key in ("tests", "excluded", "build_only") for entry in plan[key]]
        self.assertEqual(len(reported), len(set(reported)))
        self.assertEqual(discovered, set(reported))
        self.assertTrue(all(entry["reason"] for entry in plan["excluded"]))
        self.assertEqual(set(plan["build_targets"]), discovered - {entry["name"] for entry in plan["excluded"]})

    def test_cuda12_cutile_records_runtime_skip_and_mg_is_selected_by_default(self):
        plan = self.plan()
        names = {entry["name"] for entry in plan["tests"]}
        self.assertIn("CUTILE_SMOKE_TEST", names)
        self.assertIn("CLUSTER_KMEANS_MNMG_TEST", names)
        self.assertTrue(any("CUDA 13.0+" in note for note in plan["capability_notes"]))
        self.assertEqual(plan["excluded"], [])

    def test_explicitly_disabled_mg_has_reported_reasons(self):
        plan = self.plan(enable_mg=False, enable_mnmg=False)
        reasons = {entry["name"]: entry["reason"] for entry in plan["excluded"]}
        self.assertIn("NCCL", reasons["MG_C_TEST"])
        self.assertIn("UCXX", reasons["CLUSTER_KMEANS_MNMG_TEST"])

    def test_enabling_mg_allows_single_rank_tests(self):
        names = {entry["name"] for entry in self.plan(enable_mg=True, enable_mnmg=False)["tests"]}
        self.assertIn("MG_C_TEST", names)
        self.assertIn("NEIGHBORS_MG_TEST", names)
        self.assertNotIn("CLUSTER_KMEANS_MNMG_TEST", names)
        with self.assertRaisesRegex(ValueError, "enable_mg"):
            self.plan(enable_mg=False, enable_mnmg=True)

    def test_missing_configured_test_is_failure_instead_of_silent_exclusion(self):
        with self.assertRaisesRegex(ValueError, "NEW_UPSTREAM_TEST"):
            self.plan(ctest_json={"tests": [{"name": "ORDINARY_TEST"}]})
        names = self.plan()["build_targets"]
        plan = self.plan(ctest_json={"tests": [{"name": name} for name in names]})
        self.assertTrue(plan["configured_tests_validated"])

    def test_missing_source_cannot_silently_shrink_coverage(self):
        (self.source / "cpp/tests/ordinary.cu").unlink()
        with self.assertRaisesRegex(ValueError, "missing source"):
            self.plan()

    def test_header_check_detects_missing_aggregate_include(self):
        self.assertEqual(planner.check_headers(self.source)["headers_checked"], 1)
        (self.source / "c/include/cuvs/core/new.h").write_text("#pragma once\n")
        with self.assertRaisesRegex(ValueError, "cuvs/core/new.h"):
            planner.check_headers(self.source)

    def test_quick_selection_preserves_filters_commands_compile_checks_and_exclusions(self):
        selection = {"ORDINARY_TEST": "Suite.Example", "cuvs_c_headers": "*"}
        plan = self.plan(selection=selection, enable_mg=False, enable_mnmg=False)
        by_name = {entry["name"]: entry for entry in plan["tests"]}
        self.assertEqual(set(by_name), set(selection))
        self.assertEqual(by_name["ORDINARY_TEST"]["gtest_filter"], "Suite.Example")
        self.assertEqual(by_name["cuvs_c_headers"]["command"], ["bin/cuvs_c_headers"])
        self.assertEqual(plan["build_only"][0]["name"], "cuvs_c_neighbors_test")
        self.assertEqual(set(plan["build_targets"]), set(selection) | {"cuvs_c_neighbors_test"})
        reported = [entry["name"] for key in ("tests", "excluded", "build_only") for entry in plan[key]]
        self.assertCountEqual(reported, [entry["name"] for entry in planner.discover_tests(self.source)])
        self.assertTrue(all(entry["reason"] for entry in plan["excluded"]))
        configured = {"tests": [{"name": name} for name in plan["build_targets"]]}
        self.assertTrue(self.plan(selection=selection, ctest_json=configured)["configured_tests_validated"])
        with self.assertRaisesRegex(ValueError, "ORDINARY_TEST"):
            self.plan(selection=selection, ctest_json={"tests": []})

    def test_invalid_quick_selection_fails_before_building_or_silently_omitting_tests(self):
        for selection in ({}, [], {"MISSING_TEST": "*"}, {"ORDINARY_TEST": ""},
                          {"ORDINARY_TEST": None}, {"cuvs_c_neighbors_test": "*"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.plan(selection=selection)
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.plan(selection={"NEIGHBORS_MG_TEST": "*"}, enable_mg=False, enable_mnmg=False)


if __name__ == "__main__":
    unittest.main()
