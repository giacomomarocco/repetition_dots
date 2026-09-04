from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from filler.fact_eval import sglang as experiment


class PromptTests(unittest.TestCase):
    def test_official_encoder_is_used_in_chat_mode(self):
        encoder = mock.Mock(return_value="encoded")
        rendered = experiment.render_prompt(encoder, "How many?")
        self.assertEqual(rendered, "encoded")
        messages = encoder.call_args.args[0]
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertEqual(encoder.call_args.kwargs, {"thinking_mode": "chat"})

    def test_bundled_encoder_emits_deepseek_generation_suffix(self):
        encoder = experiment.load_encoder(experiment.DEFAULT_ENCODER)
        rendered = experiment.render_prompt(encoder, "How many cantos are in Inferno?")
        self.assertTrue(rendered.startswith("<｜begin▁of▁sentence｜>"))
        self.assertIn("<｜User｜>How many cantos", rendered)
        self.assertTrue(rendered.endswith("<｜Assistant｜></think>"))


class HttpTests(unittest.TestCase):
    @mock.patch("urllib.request.urlopen")
    def test_generate_request_is_greedy_and_returns_metadata(self, urlopen):
        body = {"text": "Answer: 42", "meta_info": {"completion_tokens": 3}}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(body).encode()
        urlopen.return_value = response
        text, metadata = experiment.request_generation("http://server/generate", "prompt", 10)
        self.assertEqual(text, "Answer: 42")
        self.assertEqual(metadata["completion_tokens"], 3)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["sampling_params"]["temperature"], 0)
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 8)

    @mock.patch("urllib.request.urlopen")
    def test_malformed_response_is_rejected(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"not_text": 1}'
        urlopen.return_value = response
        with self.assertRaisesRegex(ValueError, "unexpected SGLang response"):
            experiment.request_generation("http://server/generate", "prompt", 10)


if __name__ == "__main__":
    unittest.main()
