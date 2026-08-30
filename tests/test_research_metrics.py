import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app.metrics as metrics
import app.research as research


def make_success_response(
    answer="Test answer",
    key_points=None,
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
):
    if key_points is None:
        key_points = ["Point 1"]

    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            arguments=(
                '{"answer": "'
                + answer
                + '", "key_points": ["'
                + '", "'.join(key_points)
                + '"]}'
            )
        )
    )

    message = SimpleNamespace(tool_calls=[tool_call])

    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason="tool_calls",
            )
        ],
        usage=usage,
    )


class FakeRetryableError(Exception):
    pass


class TestResearchMetrics(unittest.TestCase):
    def setUp(self):
        # Prometheus counters are module-level objects, so reset their
        # values before each test to keep tests isolated.
        for metric in (
            metrics.requests_total,
            metrics.requests_success,
            metrics.requests_failed,
            metrics.retries_total,
            metrics.prompt_tokens_total,
            metrics.completion_tokens_total,
            metrics._total_tokens_counter,
            metrics.request_duration_seconds_total,
        ):
            metric._value.set(0)

    def test_successful_request_records_success_and_usage(self):
        response = make_success_response(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        def fake_create(*args, **kwargs):
            return response

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        )

        with patch.object(
            research,
            "get_client",
            return_value=fake_client,
        ):
            result = research.ask_research_question("Test question")

        current = metrics.get_metrics()

        self.assertEqual(result["question"], "Test question")
        self.assertEqual(current["requests_total"], 1)
        self.assertEqual(current["requests_success"], 1)
        self.assertEqual(current["requests_failed"], 0)
        self.assertEqual(current["retries_total"], 0)

        self.assertEqual(current["prompt_tokens_total"], 100)
        self.assertEqual(current["completion_tokens_total"], 50)
        self.assertEqual(current["total_tokens"], 150)

    def test_non_retryable_failure_records_failure_without_retry(self):
        error = RuntimeError("Something went wrong")

        with patch.object(
            research,
            "_call_llm_with_retries",
            side_effect=error,
        ):
            with patch.object(
                research,
                "get_client",
                return_value=object(),
            ):
                with self.assertRaises(RuntimeError):
                    research.ask_research_question("Test question")

        current = metrics.get_metrics()

        self.assertEqual(current["requests_total"], 1)
        self.assertEqual(current["requests_success"], 0)
        self.assertEqual(current["requests_failed"], 1)
        self.assertEqual(current["retries_total"], 0)

        self.assertEqual(current["prompt_tokens_total"], 0)
        self.assertEqual(current["completion_tokens_total"], 0)
        self.assertEqual(current["total_tokens"], 0)

    def test_three_failed_attempts_produce_two_retries(self):
        errors = [
            FakeRetryableError("failure 1"),
            FakeRetryableError("failure 2"),
            FakeRetryableError("failure 3"),
        ]

        def fake_create(*args, **kwargs):
            raise errors.pop(0)

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        )

        with patch.object(
            research,
            "RETRYABLE_EXCEPTIONS",
            (FakeRetryableError,),
        ):
            with patch.object(research.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    research._call_llm_with_retries(
                        fake_client,
                        "Test question",
                        "test-request-id",
                    )

        current = metrics.get_metrics()

        self.assertEqual(current["retries_total"], 2)

    def test_retry_then_success_records_one_retry_and_success(self):
        response = make_success_response(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        responses = [
            FakeRetryableError("temporary failure"),
            response,
        ]

        def fake_create(*args, **kwargs):
            result = responses.pop(0)

            if isinstance(result, Exception):
                raise result

            return result

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        )

        with patch.object(
            research,
            "RETRYABLE_EXCEPTIONS",
            (FakeRetryableError,),
        ):
            with patch.object(research.time, "sleep"):
                result = research._call_llm_with_retries(
                    fake_client,
                    "Test question",
                    "test-request-id",
                )

        current = metrics.get_metrics()

        self.assertEqual(result.usage.prompt_tokens, 100)

        self.assertEqual(current["requests_total"], 0)
        self.assertEqual(current["requests_success"], 0)
        self.assertEqual(current["requests_failed"], 0)

        self.assertEqual(current["retries_total"], 1)

        self.assertEqual(current["prompt_tokens_total"], 100)
        self.assertEqual(current["completion_tokens_total"], 50)
        self.assertEqual(current["total_tokens"], 150)

    def test_request_duration_is_recorded(self):
        response = make_success_response()

        def fake_create(*args, **kwargs):
            return response

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        )

        with patch.object(
            research,
            "get_client",
            return_value=fake_client,
        ):
            # Return deterministic timing values without making the test
            # depend on the exact number of internal perf_counter() calls.
            # The request starts at 100.0 and the final request duration is
            # calculated from 102.5, so the recorded duration is 2.5 seconds.
            counter_values = iter([100.0, 101.0, 102.5])

            def fake_perf_counter():
                try:
                    return next(counter_values)
                except StopIteration:
                    # If the implementation calls perf_counter() again, keep
                    # returning the final timestamp rather than failing the test.
                    return 102.5

            with patch.object(
                research.time,
                "perf_counter",
                side_effect=fake_perf_counter,
            ):
                result = research.ask_research_question("Test question")

        current = metrics.get_metrics()

        self.assertEqual(result["question"], "Test question")
        self.assertEqual(
            current["request_duration_seconds_total"],
            2.5,
        )


if __name__ == "__main__":
    unittest.main()
