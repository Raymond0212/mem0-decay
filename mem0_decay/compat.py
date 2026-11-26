"""Async Mem0 OSS-shaped adapter for add, search, and storage management."""

from .contracts import IDENTITY_FIELDS, ResolvedScope, call, finite, nonempty
from .memory import DecayMemory
from .metadata import parse_timestamp_ms


class Mem0Decay:
    """Expose Mem0 SDK arguments and dictionary results over a DecayMemory store."""

    def __init__(self, memory, *, store_id=None, tenant="default", **decay_options):
        self.tenant = nonempty(tenant, "tenant")
        self.decay = DecayMemory(
            store_id or getattr(memory, "collection_name", "mem0"),
            memory,
            **decay_options,
        )

    @classmethod
    def from_config(cls, config, **options):
        from mem0 import AsyncMemory

        return cls(AsyncMemory.from_config(config), **options)

    async def add(
        self,
        messages,
        *,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        timestamp=None,
        expiration_date=None,
        infer=True,
        memory_type=None,
        prompt=None,
        llm=None,
        event_time=None,
    ):
        """Accept benchmark/API Unix seconds through timestamp or an explicit decay event time."""
        if timestamp is not None:
            if event_time is not None:
                raise ValueError("provide either timestamp or event_time")
            if not finite(timestamp):
                raise ValueError("timestamp must be finite Unix seconds")
            event_time = timestamp * 1000
        return await self.decay.add(
            messages,
            scope=ResolvedScope(self.tenant, user_id, agent_id, run_id),
            metadata=metadata,
            event_time=event_time,
            expiration_date=expiration_date,
            infer=infer,
            memory_type=memory_type,
            prompt=prompt,
            llm=llm,
        )

    async def search(
        self,
        query,
        *,
        top_k=20,
        filters=None,
        threshold=0.1,
        rerank=False,
        explain=False,
        reference_date=None,
        show_expired=False,
    ):
        filters = dict(filters or {})
        scope = ResolvedScope(
            self.tenant,
            **{key: filters.pop(key) for key in IDENTITY_FIELDS if key in filters},
        )
        if type(top_k) is int and top_k == 0:
            nonempty(query, "query")
            return {"results": []}
        now = (
            parse_timestamp_ms(reference_date, time_zone=self.decay.default_time_zone)
            if reference_date is not None
            else None
        )
        result = await self.decay.search(
            query,
            scope=scope,
            filters=filters,
            top_k=top_k,
            threshold=threshold,
            rerank=rerank,
            explain=explain,
            show_expired=show_expired,
            reference_now_ms=now,
        )
        return {
            "results": [
                {
                    **item.native_result,
                    "id": item.id,
                    "memory": item.memory,
                    "metadata": item.metadata,
                    "score": item.adjusted_score,
                    "base_score": item.base_score,
                    "time_weight": item.time_weight,
                }
                for item in result.results
            ]
        }

    async def get(self, memory_id):
        return await call(self.decay.memory.get, memory_id)

    async def get_all(self, *, filters=None, top_k=20, show_expired=False):
        return await call(
            self.decay.memory.get_all,
            filters=filters,
            top_k=top_k,
            show_expired=show_expired,
        )

    async def history(self, memory_id):
        return await call(self.decay.memory.history, memory_id)

    async def _mutate(self, method, *args, **kwargs):
        return await self.decay.write_coordinator.run(
            self.decay.store_id, lambda: call(method, *args, **kwargs)
        )

    async def update(self, memory_id, text=None, **options):
        return await self._mutate(
            self.decay.memory.update, memory_id, text=text, **options
        )

    async def delete(self, memory_id):
        return await self._mutate(self.decay.memory.delete, memory_id)

    async def delete_all(self, **identity):
        return await self._mutate(self.decay.memory.delete_all, **identity)

    async def reset(self):
        return await self._mutate(self.decay.memory.reset)
