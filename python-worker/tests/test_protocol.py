"""Protocol tests: every message round-trips, and nothing on the wire is fatal."""

from __future__ import annotations

import json
import socket
import unittest

import harness  # noqa: F401  (puts loom_worker on sys.path)
from harness import cancel_line, dispatch_item, dispatch_line, drain_line, grant_line

from loom_worker import protocol as proto


class RoundTripTest(unittest.TestCase):
    def assert_round_trip(self, msg: object) -> None:
        line = msg.encode()  # type: ignore[attr-defined]
        self.assertTrue(line.endswith("\n"), "framing newline is part of the message")
        decoded = proto.decode_line(line)
        self.assertEqual(msg, decoded)

    def test_register(self) -> None:
        self.assert_round_trip(
            proto.Register("w-1", "tiny-gpt2", 2048, 16, 32, "cpu")
        )

    def test_heartbeat(self) -> None:
        self.assert_round_trip(proto.Heartbeat("w-1", 912, 7, 143.2, 0))

    def test_first_token(self) -> None:
        self.assert_round_trip(proto.FirstToken("r-7", 184.0))

    def test_token(self) -> None:
        self.assert_round_trip(proto.Token("r-7", " the", 12, 3))

    def test_completed_every_stop_reason(self) -> None:
        for reason in sorted(proto.STOP_REASONS):
            with self.subTest(reason=reason):
                self.assert_round_trip(proto.Completed("r-7", 48, reason))

    def test_failed_both_retryabilities(self) -> None:
        for retryable in (True, False):
            with self.subTest(retryable=retryable):
                self.assert_round_trip(proto.Failed("r-7", "boom", retryable))

    def test_preempted_every_reason(self) -> None:
        for reason in sorted(proto.PREEMPT_REASONS):
            with self.subTest(reason=reason):
                self.assert_round_trip(proto.Preempted("r-7", 31, reason))

    def test_grow_request(self) -> None:
        self.assert_round_trip(proto.GrowRequest("r-7", 4, 112))

    def test_rejected(self) -> None:
        self.assert_round_trip(proto.Rejected("r-7", proto.REJECT_BACKPRESSURE))

    def test_grant(self) -> None:
        self.assertEqual(
            proto.decode_line(grant_line("r-7", 4)), proto.Grant("r-7", 4)
        )

    def test_dispatch(self) -> None:
        line = dispatch_line(
            [
                dispatch_item("r-7", prompt="hello", max_output_tokens=128, reserved_blocks=12),
                dispatch_item("r-8", prompt="world", max_output_tokens=16, reserved_blocks=3),
            ]
        )
        decoded = proto.decode_line(line)
        self.assertEqual(
            decoded,
            proto.Dispatch(
                batch=(
                    proto.DispatchItem("r-7", "hello", 128, 12),
                    proto.DispatchItem("r-8", "world", 16, 3),
                )
            ),
        )

    def test_dispatch_with_empty_batch(self) -> None:
        self.assertEqual(proto.decode_line(dispatch_line([])), proto.Dispatch(batch=()))

    def test_cancel(self) -> None:
        self.assertEqual(proto.decode_line(cancel_line("r-7")), proto.Cancel("r-7"))

    def test_drain(self) -> None:
        self.assertEqual(proto.decode_line(drain_line()), proto.Drain())

    def test_exact_wire_shape_of_the_new_messages(self) -> None:
        self.assertEqual(
            json.loads(proto.GrowRequest("r-7", 4, 112).encode()),
            {"type": "grow_request", "request_id": "r-7", "want_blocks": 4, "free_blocks": 112},
        )
        self.assertEqual(
            json.loads(proto.Rejected("r-7").encode()),
            {"type": "rejected", "request_id": "r-7", "reason": "backpressure"},
        )
        self.assertEqual(
            json.loads(proto.Preempted("r-7", 31, proto.PREEMPT_CACHE_EXHAUSTED).encode()),
            {"type": "preempted", "request_id": "r-7", "generated": 31, "reason": "cache_exhausted"},
        )

    def test_exact_wire_shape_of_an_encoded_message(self) -> None:
        # The Go side matches on these names; a round trip through our own decoder
        # would pass even if both halves agreed on the wrong spelling.
        obj = json.loads(proto.Token("r-7", " the", 12, 3).encode())
        self.assertEqual(
            obj,
            {"type": "token", "request_id": "r-7", "token": " the", "generated": 12, "blocks": 3},
        )


class ToleranceTest(unittest.TestCase):
    def test_unknown_type_is_skipped_not_fatal(self) -> None:
        with self.assertLogs("loom_worker.protocol", level="INFO") as logs:
            self.assertIsNone(proto.decode_line('{"type":"rebalance","shards":4}'))
        self.assertTrue(any("unknown message type" in line for line in logs.output))

    def test_malformed_json_is_skipped_not_fatal(self) -> None:
        with self.assertLogs("loom_worker.protocol", level="WARNING"):
            self.assertIsNone(proto.decode_line('{"type":"cancel", "request_id":'))

    def test_non_object_json_is_skipped(self) -> None:
        with self.assertLogs("loom_worker.protocol", level="WARNING"):
            self.assertIsNone(proto.decode_line("[1,2,3]"))

    def test_missing_type_is_skipped(self) -> None:
        with self.assertLogs("loom_worker.protocol", level="WARNING"):
            self.assertIsNone(proto.decode_line('{"request_id":"r-1"}'))

    def test_blank_line_is_skipped_silently(self) -> None:
        self.assertIsNone(proto.decode_line("\n"))

    def test_unknown_fields_are_ignored(self) -> None:
        line = json.dumps(
            {"type": "cancel", "request_id": "r-7", "priority": 9, "future_field": {"a": 1}}
        )
        self.assertEqual(proto.decode_line(line), proto.Cancel("r-7"))

    def test_missing_fields_decode_to_zero_values(self) -> None:
        # A short message is still a message; refusing it would make a control-plane
        # version skew fatal, which is what PROTOCOL.md forbids.
        self.assertEqual(proto.decode_line('{"type":"token"}'), proto.Token("", "", 0, 0))

    def test_unknown_stop_reason_decodes(self) -> None:
        # Strict outbound, tolerant inbound: a worker must not invent a stop reason,
        # but it must not die because a peer did.
        decoded = proto.decode_line('{"type":"completed","request_id":"r","generated":1,"stop_reason":"wat"}')
        self.assertEqual(decoded, proto.Completed("r", 1, "wat"))

    def test_unknown_preempt_reason_decodes(self) -> None:
        decoded = proto.decode_line(
            '{"type":"preempted","request_id":"r","generated":2,"reason":"from_the_future"}'
        )
        self.assertEqual(decoded, proto.Preempted("r", 2, "from_the_future"))

    def test_grant_with_a_missing_amount_decodes_to_zero(self) -> None:
        # Zero is then refused by the engine, which is the right place for the policy:
        # decoding stays mechanical and tolerant.
        self.assertEqual(proto.decode_line('{"type":"grant","request_id":"r"}'), proto.Grant("r", 0))

    def test_dispatch_batch_entry_of_wrong_shape_is_skipped(self) -> None:
        line = json.dumps({"type": "dispatch", "batch": ["nonsense", dispatch_item("r-1")]})
        with self.assertLogs("loom_worker.protocol", level="WARNING"):
            decoded = proto.decode_line(line)
        assert isinstance(decoded, proto.Dispatch)
        self.assertEqual([i.request_id for i in decoded.batch], ["r-1"])


class StrictEncoderTest(unittest.TestCase):
    def test_typo_in_field_name_is_loud(self) -> None:
        with self.assertRaises(proto.ProtocolError) as ctx:
            proto.encode_payload(
                proto.TYPE_TOKEN,
                {"request_id": "r-7", "token": " the", "generated": 12, "block": 3},
            )
        self.assertIn("unknown field", str(ctx.exception))

    def test_missing_field_is_loud(self) -> None:
        with self.assertRaises(proto.ProtocolError) as ctx:
            proto.encode_payload(proto.TYPE_TOKEN, {"request_id": "r-7", "token": " the"})
        self.assertIn("missing field", str(ctx.exception))

    def test_wrong_type_is_loud(self) -> None:
        with self.assertRaises(proto.ProtocolError):
            proto.encode_payload(
                proto.TYPE_TOKEN,
                {"request_id": "r-7", "token": " the", "generated": "12", "blocks": 3},
            )

    def test_bool_in_a_count_field_is_loud(self) -> None:
        # json.dumps(True) is `true`, which the Go side cannot unmarshal into an int,
        # and bool passes isinstance(x, int) in Python.
        with self.assertRaises(proto.ProtocolError):
            proto.encode_payload(
                proto.TYPE_TOKEN,
                {"request_id": "r-7", "token": " the", "generated": True, "blocks": 3},
            )

    def test_unlisted_stop_reason_is_loud(self) -> None:
        with self.assertRaises(proto.ProtocolError):
            proto.Completed("r-7", 3, "finished").encode()

    def test_unlisted_preempt_reason_is_loud(self) -> None:
        # The control plane routes on this value; an unbucketed one is a metrics hole
        # in the one signal that says whether to add workers or fix the estimator.
        with self.assertRaises(proto.ProtocolError):
            proto.Preempted("r-7", 3, "out_of_memory").encode()

    def test_unlisted_reject_reason_is_loud(self) -> None:
        with self.assertRaises(proto.ProtocolError):
            proto.Rejected("r-7", "too_busy").encode()

    def test_typo_in_a_grow_request_field_is_loud(self) -> None:
        with self.assertRaises(proto.ProtocolError):
            proto.encode_payload(
                proto.TYPE_GROW_REQUEST,
                {"request_id": "r-7", "want_blocks": 4, "free": 112},
            )

    def test_grant_cannot_be_encoded_as_outbound(self) -> None:
        # grant is control plane to worker; a worker sending one is a bug, not a
        # message, and the schema is where that is caught.
        with self.assertRaises(proto.ProtocolError):
            proto.encode_payload(proto.TYPE_GRANT, {"request_id": "r-7", "extra_blocks": 4})

    def test_inbound_type_cannot_be_encoded_as_outbound(self) -> None:
        with self.assertRaises(proto.ProtocolError):
            proto.encode_payload(proto.TYPE_DISPATCH, {"batch": []})

    def test_integers_are_accepted_where_floats_are_allowed(self) -> None:
        obj = json.loads(proto.Heartbeat("w", 1, 2, 3, 0).encode())
        self.assertEqual(obj["tokens_per_second"], 3)


class FramingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a, self.b = socket.socketpair()
        self.addCleanup(self.a.close)
        self.addCleanup(self.b.close)

    def test_lines_split_across_reads_are_reassembled(self) -> None:
        reader = proto.FramedConnection(self.b)
        # One message in two pieces, with the newline only in the second piece.
        line = proto.Token("r-1", " the", 1, 1).encode()
        self.a.sendall(line[:7].encode())
        self.a.sendall(line[7:].encode())
        self.assertEqual(
            proto.decode_line(reader.read_line() or ""), proto.Token("r-1", " the", 1, 1)
        )

    def test_multiple_messages_in_one_read_are_separated(self) -> None:
        reader = proto.FramedConnection(self.b)
        blob = cancel_line("r-1") + drain_line() + cancel_line("r-2")
        self.a.sendall(blob.encode())
        self.a.shutdown(socket.SHUT_WR)
        self.assertEqual(
            list(reader.messages()),
            [proto.Cancel("r-1"), proto.Drain(), proto.Cancel("r-2")],
        )

    def test_unknown_type_in_a_stream_does_not_stop_iteration(self) -> None:
        reader = proto.FramedConnection(self.b)
        blob = '{"type":"from_the_future"}\n' + "not json at all\n" + cancel_line("r-9")
        self.a.sendall(blob.encode())
        self.a.shutdown(socket.SHUT_WR)
        with self.assertLogs("loom_worker.protocol", level="INFO"):
            self.assertEqual(list(reader.messages()), [proto.Cancel("r-9")])

    def test_trailing_fragment_at_eof_is_discarded(self) -> None:
        reader = proto.FramedConnection(self.b)
        self.a.sendall(b'{"type":"cancel"')
        self.a.shutdown(socket.SHUT_WR)
        with self.assertLogs("loom_worker.protocol", level="WARNING"):
            self.assertEqual(list(reader.messages()), [])


if __name__ == "__main__":
    unittest.main()
