"""
Unit tests for bio_agent.

These use only the standard library (unittest + unittest.mock). Every network
call is mocked, so the suite runs offline with no API key:

    python -m unittest discover -s tests -v

What's covered:
  * the data tools (mygene_query, pubmed_search), incl. their empty-result paths
  * the keyword grader (score_answer)
  * the multi-step agent loop (run_agent) with a faked LLM client and mocked HTTP
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Make the repo root importable regardless of how the tests are launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bio_agent  # noqa: E402


def _response(json_payload):
    """Build a fake `requests` response returning the given JSON."""
    resp = Mock()
    resp.json.return_value = json_payload
    resp.raise_for_status.return_value = None
    return resp


class TestMyGeneQuery(unittest.TestCase):
    @patch("bio_agent.requests.get")
    def test_returns_first_hit(self, mock_get):
        mock_get.return_value = _response(
            {"hits": [{"symbol": "mecA", "name": "penicillin-binding protein 2a"}]}
        )
        result = bio_agent.mygene_query("mecA")
        self.assertEqual(result["symbol"], "mecA")
        self.assertIn("name", result)

    @patch("bio_agent.requests.get")
    def test_no_hits_returns_error(self, mock_get):
        mock_get.return_value = _response({"hits": []})
        result = bio_agent.mygene_query("not_a_real_gene")
        self.assertIn("error", result)


class TestPubmedSearch(unittest.TestCase):
    @patch("bio_agent.time.sleep", return_value=None)  # don't actually wait
    @patch("bio_agent.requests.get")
    def test_parses_papers(self, mock_get, _sleep):
        mock_get.side_effect = [
            _response({"esearchresult": {"idlist": ["111", "222"]}}),
            _response(
                {
                    "result": {
                        "111": {
                            "title": "Paper one",
                            "fulljournalname": "Journal A",
                            "pubdate": "2020 Jan",
                        },
                        "222": {
                            "title": "Paper two",
                            "fulljournalname": "Journal B",
                            "pubdate": "2021 Mar",
                        },
                    }
                }
            ),
        ]
        papers = bio_agent.pubmed_search("mecA resistance")
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0]["pmid"], "111")
        self.assertEqual(papers[0]["year"], "2020")
        self.assertTrue(papers[0]["url"].endswith("/111/"))

    @patch("bio_agent.requests.get")
    def test_no_ids_returns_empty(self, mock_get):
        mock_get.return_value = _response({"esearchresult": {"idlist": []}})
        self.assertEqual(bio_agent.pubmed_search("nonsense query xyz"), [])


class TestScoreAnswer(unittest.TestCase):
    def test_all_keywords_present_passes(self):
        hits, ok = bio_agent.score_answer(
            "Found in Staphylococcus aureus, conferring methicillin resistance.",
            ["staphylococcus", "methicillin"],
        )
        self.assertTrue(ok)
        self.assertEqual(len(hits), 2)

    def test_missing_keyword_fails(self):
        hits, ok = bio_agent.score_answer(
            "It is found in Staphylococcus aureus.",
            ["staphylococcus", "methicillin"],
        )
        self.assertFalse(ok)
        self.assertEqual(hits, ["staphylococcus"])

    def test_matching_is_case_insensitive(self):
        _hits, ok = bio_agent.score_answer("VANCOMYCIN resistance", ["vancomycin"])
        self.assertTrue(ok)


# --- Fakes for the agent-loop test ----------------------------------------- #
class FakeBlock:
    def __init__(self, type, text=None, name=None, input=None, id=None):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class FakeResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


class TestRunAgentLoop(unittest.TestCase):
    @patch("bio_agent.requests.get")
    @patch("bio_agent.get_client")
    def test_calls_tool_then_returns_final_answer(self, mock_get_client, mock_get):
        # Turn 1: model asks to call mygene_query. Turn 2: model gives the answer.
        fake_client = FakeClient(
            [
                FakeResponse(
                    "tool_use",
                    [
                        FakeBlock(
                            "tool_use",
                            name="mygene_query",
                            input={"symbol": "mecA"},
                            id="tool_1",
                        )
                    ],
                ),
                FakeResponse(
                    "end_turn",
                    [FakeBlock("text", text="mecA is found in Staphylococcus (MyGene.info).")],
                ),
            ]
        )
        mock_get_client.return_value = fake_client
        # The tool call inside the loop hits MyGene -> mock that HTTP call.
        mock_get.return_value = _response({"hits": [{"symbol": "mecA"}]})

        answer = bio_agent.run_agent("What is mecA?", verbose=False)

        self.assertIn("Staphylococcus", answer)
        # The model was called twice: once to request the tool, once to answer.
        self.assertEqual(len(fake_client.messages.calls), 2)
        # The second call must have fed a tool_result back to the model.
        second_call_messages = fake_client.messages.calls[1]["messages"]
        tool_result_sent = any(
            isinstance(m["content"], list)
            and any(
                isinstance(part, dict) and part.get("type") == "tool_result"
                for part in m["content"]
            )
            for m in second_call_messages
        )
        self.assertTrue(tool_result_sent)

    @patch("bio_agent.get_client")
    def test_stops_at_max_steps(self, mock_get_client):
        # Model keeps asking for a tool forever; loop must stop gracefully.
        always_tool = FakeResponse(
            "tool_use",
            [FakeBlock("tool_use", name="mygene_query", input={"symbol": "x"}, id="t")],
        )

        class LoopingMessages(FakeMessages):
            def create(self, **kwargs):
                self.calls.append(kwargs)
                return always_tool

        client = FakeClient([])
        client.messages = LoopingMessages([])
        mock_get_client.return_value = client

        with patch("bio_agent.requests.get", return_value=_response({"hits": []})):
            answer = bio_agent.run_agent("loop forever", max_steps=3, verbose=False)

        self.assertIn("max steps", answer.lower())
        self.assertEqual(len(client.messages.calls), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
