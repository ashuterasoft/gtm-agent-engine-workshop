import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

package = ModuleType("gtm_agent")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules["gtm_agent"] = package

with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    from gtm_agent import gtm_agent as agent_module
    from gtm_agent.gtm_agent import SYSTEM_PROMPT, data_service, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {}})
        self.from_rep = {"name": "Ava Mills", "email": "ava@example.com"}

    def send(self, prospect):
        return send_prospect_email.func(
            prospect,
            "Hello",
            "A note",
            self.runtime,
            from_rep=self.from_rep,
        )

    def test_disqualified_prospect_is_blocked_without_message_id(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Priya Nair",
            "email": "model@example.com",
        }
        with patch.object(
            data_service,
            "get_prospect_record",
            wraps=data_service.get_prospect_record,
        ) as get_record, patch.object(agent_module.uuid, "uuid4") as generate_id:
            result = self.send(prospect)

        self.assertEqual(
            result,
            {
                "status": "blocked",
                "reason": "Prospect is disqualified; email not sent.",
            },
        )
        get_record.assert_called_once_with("LEAD-50001")
        generate_id.assert_not_called()

    def test_qualified_prospect_can_be_emailed(self):
        record = {
            "prospect_id": "LEAD-12345",
            "name": "Alex Rivera",
            "email": "alex@example.com",
            "disqualified": False,
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = self.send({"prospect_id": "LEAD-12345"})

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "alex@example.com")

    def test_missing_record_fails_without_message_id(self):
        prospect = {"prospect_id": "LEAD-404", "email": "model@example.com"}
        with patch.object(data_service, "get_prospect_record", return_value=None), patch.object(
            agent_module.uuid, "uuid4"
        ) as generate_id:
            result = self.send(prospect)

        self.assertEqual(result, {"status": "failed", "error": "Prospect record not found."})
        generate_id.assert_not_called()

    def test_model_supplied_email_is_ignored(self):
        record = {
            "prospect_id": "LEAD-12345",
            "name": "Alex Rivera",
            "email": "alex@example.com",
            "disqualified": False,
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = self.send(
                {
                    "prospect_id": "LEAD-12345",
                    "name": "Wrong Name",
                    "email": "wrong@example.com",
                }
            )

        self.assertEqual(result["to"], "alex@example.com")
        self.assertEqual(result["to_name"], "Alex Rivera")

    def test_agent_prompt_explains_blocked_sends_to_rep(self):
        self.assertIn(
            "tell the rep the email was not sent and explain why",
            SYSTEM_PROMPT,
        )
