"""The REST contract used by Option B on the benchmark's main branch."""

import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

HAS_SERVER = all(
    importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx", "mem0")
)


@unittest.skipUnless(HAS_SERVER, "install the server extra")
class OSSServerTest(unittest.TestCase):
    def test_option_b_ingestion_ranking_dates_and_cleanup(self):
        from fastapi.testclient import TestClient

        from mem0_decay.server import app

        native = MagicMock()
        native.collection_name = "test"
        native.__aenter__.return_value = native
        stored = []

        def add(messages, *, user_id, metadata, infer, prompt):
            row = {
                "id": str(len(stored)),
                "memory": messages[0]["content"],
                "score": 0.8,
                "metadata": metadata,
                "created_at": "2026-01-01T00:00:00Z",
            }
            stored.append(row)
            return {"results": [row]}

        native.add.side_effect = add
        native.search.return_value = {"results": stored}
        with (
            patch.dict(
                os.environ, {"MEM0_DECAY_REFERENCE_DATE": "1970-01-31T00:00:00Z"}
            ),
            patch("mem0_decay.server._open_sdk", return_value=native),
            TestClient(app) as client,
        ):
            health = client.get("/health").json()
            self.assertEqual(health["memory"], "mem0-decay")
            for timestamp, text in [(0, "older"), (2592000, "newer")]:
                response = client.post(
                    "/memories",
                    json={
                        "messages": [{"role": "user", "content": text}],
                        "user_id": "alice",
                        "custom_instructions": "Remember the facts",
                        "timestamp": timestamp,
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["results"][0]["memory"], text)
            self.assertEqual(
                native.add.call_args.kwargs["prompt"], "Remember the facts"
            )
            response = client.post(
                "/search",
                json={
                    "query": "facts",
                    "user_id": "alice",
                    "limit": 200,
                    "rerank": True,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            newer, older = response.json()["results"]
            self.assertEqual([newer["memory"], older["memory"]], ["newer", "older"])
            self.assertAlmostEqual(newer["score"], 0.8)
            self.assertAlmostEqual(older["score"], 0.8 * 0.975)
            self.assertEqual(older["base_score"], 0.8)
            self.assertEqual(older["created_at"], "1970-01-01T00:00:00.000Z")
            self.assertEqual(
                native.search.call_args.kwargs["filters"], {"user_id": "alice"}
            )
            self.assertEqual(native.search.call_args.kwargs["top_k"], 200)
            self.assertTrue(native.search.call_args.kwargs["rerank"])
            self.assertEqual(client.delete("/memories").status_code, 422)
            native.delete_all.assert_not_called()
            self.assertEqual(
                client.delete("/memories", params={"user_id": "alice"}).status_code, 200
            )
            native.delete_all.assert_called_once_with(user_id="alice")
        native.__aexit__.assert_awaited_once()
