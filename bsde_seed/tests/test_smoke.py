# 10 LoC sanity test
import unittest

# Attempt to import the main components
try:
    from bsde_seed.bsde_dsgE.core.solver import Solver
    from bsde_seed.bsde_dsgE.models.ct_lucas import CTLucas
    imports_successful = True
except ImportError as e:
    imports_successful = False
    import_error_message = str(e)

class TestSmoke(unittest.TestCase):
    def test_imports(self):
        """
        Tests that essential modules and classes can be imported.
        """
        self.assertTrue(imports_successful,
                        f"Failed to import modules: {import_error_message if not imports_successful else ''}")

    def test_instantiation(self):
        """
        Tests basic instantiation of core classes.
        """
        if not imports_successful:
            self.skipTest("Skipping instantiation test due to import failures.")

        try:
            model = CTLucas()
            solver = Solver(model)
            self.assertIsNotNone(model, "Model instantiation failed.")
            self.assertIsNotNone(solver, "Solver instantiation failed.")
        except Exception as e:
            self.fail(f"Instantiation raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
