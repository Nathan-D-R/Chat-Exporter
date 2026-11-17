import unittest

from chat_exporter.parser import ChatRequest, ChatSession, ResponsePart, ToolCall
from chat_exporter.renderer import render_session


class RendererTests(unittest.TestCase):
    def test_condensed_render_keeps_prompts_and_assistant_prose_only(self):
        session = ChatSession(
            session_id="session-1",
            title="Investigation",
            created_ms=1_775_178_266_000,
            model_id="test-model",
            provider="copilot-cli",
            workspace="demo-tool",
            requests=[
                ChatRequest(
                    request_id="r1",
                    user_text="Review the suspicious transactions.",
                    response_parts=[
                        ResponsePart(
                            "text",
                            text="I reviewed the transactions and found two high-value cases.",
                        ),
                        ResponsePart(
                            "tool_call",
                            tool=ToolCall(
                                name="db_query",
                                output="sensitive raw transaction output",
                            ),
                        ),
                    ],
                )
            ],
        )

        content = render_session(session, condensed=True)

        self.assertIn("Review the suspicious transactions.", content)
        self.assertIn("two high-value cases", content)
        self.assertNotIn("db_query", content)
        self.assertNotIn("sensitive raw transaction output", content)
        self.assertNotIn("Tool calls:", content)
        self.assertNotIn("Files changed:", content)

    def test_condensed_render_bounds_assistant_prose(self):
        session = ChatSession(
            session_id="session-1",
            title="Investigation",
            created_ms=1_775_178_266_000,
            model_id="test-model",
            requests=[
                ChatRequest(
                    request_id="r1",
                    user_text="Review it.",
                    response_parts=[
                        ResponsePart("text", text="start " + ("middle " * 100) + " end"),
                    ],
                )
            ],
        )

        content = render_session(session, condensed=True, max_assistant_chars=40)

        self.assertIn("[assistant response condensed]", content)
        self.assertLessEqual(
            len(content.split("**Assistant**\n\n", 1)[1].split("\n\n", 1)[0]),
            40,
        )
