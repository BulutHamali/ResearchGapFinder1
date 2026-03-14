import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Async token bucket rate limiter.

    Allows up to `rate` requests per `period` seconds.
    """

    def __init__(self, rate: float, period: float = 1.0, name: str = "limiter"):
        self.rate = rate          # tokens per period
        self.period = period      # period in seconds
        self.name = name
        self._tokens = rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens based on elapsed time
            self._tokens = min(
                self.rate,
                self._tokens + elapsed * (self.rate / self.period),
            )
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return

            # Need to wait
            wait_time = (tokens - self._tokens) * (self.period / self.rate)
            logger.debug(f"RateLimiter '{self.name}': waiting {wait_time:.2f}s")
        # Release the lock while sleeping so other coroutines can queue up
        await asyncio.sleep(wait_time)
        async with self._lock:
            self._tokens = max(0.0, self._tokens - tokens)

    def acquire_sync(self, tokens: float = 1.0) -> None:
        """Synchronous version: sleep until a token is available."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.rate,
            self._tokens + elapsed * (self.rate / self.period),
        )
        self._last_refill = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return

        wait_time = (tokens - self._tokens) * (self.period / self.rate)
        logger.debug(f"RateLimiter '{self.name}': sleeping {wait_time:.2f}s")
        time.sleep(wait_time)
        self._tokens = max(0.0, self._tokens - tokens)


# Default rate limiters
# NCBI allows 3 requests/second without API key, 10 with API key
pubmed_limiter = RateLimiter(rate=3, period=1.0, name="pubmed")

# Groq has generous limits; use conservative 10 req/s
groq_limiter = RateLimiter(rate=10, period=1.0, name="groq")
