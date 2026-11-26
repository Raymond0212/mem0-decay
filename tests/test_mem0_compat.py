import unittest

from mem0_decay import Mem0Decay

DAY = 86_400


class SDKFixture:
    def __init__(self):
        self.rows = []
        self.search_options = None
        self.prompt = None

    async def add(self, messages, *, user_id, metadata, infer=True, prompt=None):
        self.prompt = prompt
        item = {
            "id": str(len(self.rows)),
            "memory": messages,
            "user_id": user_id,
            "metadata": metadata,
            "score": 0.8,
            "score_details": {"semantic": 0.8},
        }
        self.rows.append(item)
        return {"results": [item]}

    async def search(
        self, query, *, filters, top_k, threshold, rerank, explain, show_expired
    ):
        self.search_options = filters, top_k, explain
        return {"results": self.rows[:top_k]}


class Mem0CompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_timestamp_filters_and_result_shape(self):
        sdk = SDKFixture()
        mem = Mem0Decay(sdk, alpha=1)
        await mem.add("old", user_id="alice", timestamp=0, prompt="Extract names")
        self.assertEqual(sdk.prompt, "Extract names")
        await mem.add("recent", user_id="alice", timestamp=30 * DAY)
        result = await mem.search(
            "query",
            filters={"user_id": "alice", "kind": "fact"},
            top_k=2,
            explain=True,
            reference_date=30 * DAY * 1000,
        )
        self.assertEqual(
            sdk.search_options, ({"user_id": "alice", "kind": "fact"}, 8, True)
        )
        first, second = result["results"]
        self.assertEqual([first["memory"], second["memory"]], ["recent", "old"])
        self.assertEqual(second["metadata"]["decay_time_end"], 0)
        self.assertEqual(first["metadata"]["decay_time_end"], 30 * DAY * 1000)
        self.assertAlmostEqual(second["score"], 0.52)
        self.assertEqual(second["base_score"], 0.8)
        self.assertEqual(second["score_details"], {"semantic": 0.8})
