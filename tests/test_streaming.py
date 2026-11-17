"""Regression coverage for incremental MCP discovery and bounded ranking."""
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from chat_exporter.parser import ChatRequest, ChatSession
from chat_exporter.providers import DiscoveredSession, iter_discover, parse_claude, parse_codex, parse_copilot_cli
from chat_exporter.search import read_session, search_discovered, search_history


def item(index):
    session = ChatSession(str(index), 'Same title', index % 7 + 1, '',
                          [ChatRequest(user_text='needle ' * (index % 3 + 1))],
                          'codex', 'demo')
    return DiscoveredSession('codex', 'demo', Path(f'{index}.jsonl'), session)


class StreamingTests(unittest.TestCase):
    def test_discovery_does_not_parse_until_consumed(self):
        def adapter(root):
            yield item(0)
            raise AssertionError('unrequested session parsed')
        with patch('chat_exporter.providers.discover_codex', adapter):
            stream = iter_discover({'codex'})
            self.assertEqual(next(stream).session.session_id, '0')
            stream.close()

    def test_read_stops_at_match(self):
        def adapter(root):
            yield item(0)
            raise AssertionError('read parsed beyond the requested session')
        with patch('chat_exporter.providers.discover_codex', adapter):
            result = read_session('codex', '0')
        self.assertEqual(result['session']['session_id'], '0')
        self.assertIn('needle', result['content'])

    def test_search_releases_previous_sessions(self):
        references = []
        def adapter(root):
            for index in range(100):
                session_item = item(index)
                references.append(weakref.ref(session_item.session))
                self.assertLessEqual(sum(ref() is not None for ref in references), 3)
                yield session_item
        with patch('chat_exporter.providers.discover_codex', adapter):
            result = search_history('needle', providers=['codex'], limit=3)
        self.assertEqual(result['matching_sessions'], 100)
        self.assertEqual(len(result['results']), 3)
        self.assertTrue(all(ref() is None for ref in references))

    def test_bounded_ranking_matches_full_sort_including_ties(self):
        for query, sort_by in [('needle', 'relevance'), ('needle', 'recent'), ('', 'recent')]:
            with self.subTest(query=query, sort_by=sort_by):
                hits = [search_discovered([item(i)], query, sort_by=sort_by)['results'][0]
                        for i in range(200)]
                # ISO timestamps sort chronologically; stable passes preserve exact ties.
                hits.sort(key=lambda h: h['title'].casefold())
                hits.sort(key=lambda h: h['created_at'], reverse=True)
                if sort_by == 'relevance':
                    hits.sort(key=lambda h: h['score'], reverse=True)
                result = search_discovered((item(i) for i in range(200)), query,
                                           limit=5, sort_by=sort_by)
                self.assertEqual(result['results'], hits[:5])
                self.assertEqual(result['matching_sessions'], 200)

    def test_jsonl_parsers_stream_and_ignore_malformed_records(self):
        cases = [
            (parse_claude, [{'type': 'user', 'cwd': '/tmp/demo',
                            'message': {'role': 'user', 'content': 'hello'}}]),
            (parse_codex, [{'type': 'session_meta', 'payload': {'id': 'test'}},
                           {'type': 'response_item', 'payload': {'type': 'message',
                            'role': 'user', 'content': 'hello'}}]),
            (lambda p: parse_copilot_cli(p, {}),
             [{'type': 'user.message', 'data': {'content': 'hello'}}]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            for parser, records in cases:
                path.write_text('\nnot-json\nnull\n[]\n' + '\n'.join(map(json.dumps, records)))
                with patch.object(Path, 'read_text', side_effect=AssertionError('whole-file read')):
                    session = parser(path)
                self.assertEqual(session.requests[0].user_text, 'hello')
