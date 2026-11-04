"""Offline behavior tests; no API key, network, or OpenAI package required."""

import contextlib
import importlib.util
import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Provide only the missing SDK interface for offline tests. Never fake live output.
if importlib.util.find_spec("openai") is None:
    sdk = types.ModuleType("openai")
    sdk.OpenAI = MagicMock()
    sdk.OpenAIError = type("OpenAIError", (Exception,), {})
    sys.modules["openai"] = sdk

import project_05 as project


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.output)
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)

    def test_history_accumulates(self):
        snapshots = []

        def reply(messages):
            snapshots.append(list(messages))
            return "Test response"

        with patch("builtins.input", side_effect=["My name is Jordan", "What is my name?", "quit"]), \
                patch.object(project, "is_safe", return_value=True), \
                patch.object(project, "get_completion", side_effect=reply):
            project.run_chatbot()
        self.assertEqual([len(m) for m in snapshots], [2, 4])
        self.assertEqual(snapshots[1][1]["content"], "My name is Jordan")
        self.assertEqual(snapshots[1][2]["role"], "assistant")

    def test_unsafe_feature_data_is_not_sent(self):
        with patch("builtins.input", side_effect=["resume", "unsafe text", "DONE", "quit"]), \
                patch.object(project, "is_safe", side_effect=[True, False]) as moderation, \
                patch.object(project, "rewrite_bullets") as rewrite:
            project.run_chatbot()
        rewrite.assert_not_called()
        self.assertEqual(moderation.call_args_list[1].args, ("unsafe text",))

    def test_feature_routes_and_history(self):
        with patch("builtins.input", side_effect=["resume", "Made reports", "DONE",
                                                 "cover letter", "Analyst", "Teacher",
                                                 "Make it shorter", "exit"]), \
                patch.object(project, "is_safe", return_value=True), \
                patch.object(project, "rewrite_bullets", return_value=[{"original": "Made reports", "improved": "Prepared reports"}]) as rewrite, \
                patch.object(project, "generate_cover_letter", return_value="Opening") as cover, \
                patch.object(project, "get_completion", return_value="Short opening") as chat:
            project.run_chatbot()
        rewrite.assert_called_once_with(["Made reports"])
        cover.assert_called_once_with("Analyst", "Teacher")
        self.assertEqual(len(chat.call_args.args[0]), 6)

    def test_quit_inside_feature(self):
        with patch("builtins.input", side_effect=["cover letter", "quit"]), \
                patch.object(project, "is_safe", return_value=True), \
                patch.object(project, "generate_cover_letter") as cover:
            project.run_chatbot()
        cover.assert_not_called()
        self.assertIn("Good luck", self.output.getvalue())

    def test_empty_and_flagged_turns_do_not_call_model(self):
        with patch("builtins.input", side_effect=["", "flagged", "quit"]), \
                patch.object(project, "is_safe", return_value=False) as moderation, \
                patch.object(project, "get_completion") as chat:
            project.run_chatbot()
        moderation.assert_called_once_with("flagged")
        chat.assert_not_called()

    def test_bullet_json_and_original_validation(self):
        expected = [{"original": "Made reports", "improved": "Prepared management reports"}]
        with patch.object(project, "get_completion", return_value=json.dumps(expected)):
            self.assertEqual(project.rewrite_bullets(["Made reports"]), expected)
        for raw in ("not JSON", "{}", '[{"original":"Changed input","improved":"Text"}]'):
            with self.subTest(raw=raw), patch.object(project, "get_completion", return_value=raw):
                with self.assertRaises(ValueError):
                    project.rewrite_bullets(["Made reports"])

    def test_moderation_returns_boolean(self):
        for flagged in (False, True):
            result = types.SimpleNamespace(results=[types.SimpleNamespace(flagged=flagged)])
            with patch.object(project, "client") as client:
                client.moderations.create.return_value = result
                self.assertIs(project.is_safe("test"), not flagged)
                client.moderations.create.assert_called_once_with(
                    model="omni-moderation-latest", input="test")

    def test_api_error_allows_next_turn(self):
        with patch("builtins.input", side_effect=["Hello", "Try again", "quit"]), \
                patch.object(project, "is_safe", return_value=True), \
                patch.object(project, "get_completion", side_effect=[ValueError("bad output"), "OK"]) as chat:
            project.run_chatbot()
        self.assertEqual(len(chat.call_args.args[0]), 2)


if __name__ == "__main__":
    unittest.main()
