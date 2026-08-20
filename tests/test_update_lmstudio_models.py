import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "update_lmstudio_models.py"
SPEC = importlib.util.spec_from_file_location("update_lmstudio_models", SCRIPT_PATH)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class ContextResolutionTests(unittest.TestCase):
    def test_loaded_variant_context_is_associated_with_model_key(self):
        response = {
            "models": [
                {
                    "key": "gemma-4-12b-it@iq3_xs",
                    "max_context_length": 131072,
                    "loaded_instances": [],
                },
                {
                    "key": "gemma-4-12b-it@iq4_xs",
                    "max_context_length": 131072,
                    "loaded_instances": [
                        {
                            "id": "gemma-4-12b-it",
                            "config": {"context_length": 38400},
                        }
                    ],
                },
            ]
        }

        server_models, loaded_instances, _ = sync.parse_models_response(response)

        self.assertEqual(loaded_instances, {"gemma-4-12b-it@iq4_xs": 38400})
        self.assertEqual(
            sync.get_context_for_model(
                "http://lmstudio:1234",
                "gemma-4-12b-it@iq4_xs",
                server_models,
                {},
                loaded_instances,
            ),
            38400,
        )
        self.assertEqual(
            sync.get_context_for_model(
                "http://lmstudio:1234",
                "gemma-4-12b-it@iq3_xs",
                server_models,
                {},
                loaded_instances,
            ),
            131072,
        )


if __name__ == "__main__":
    unittest.main()
