"""BDD steps (spec 007 AC-3).

The feature files describe behaviour a reviewer would ask about; these steps
drive the same HTTP surface the integration tests use.
"""
from __future__ import annotations

import io

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("../features")


@pytest.fixture
def context():
    return {}


def _upload(client, name: str, body: str, tenant: str = "tester"):
    return client.post(
        "/api/upload",
        files={"file": (name, io.BytesIO(body.encode()), "text/plain")},
        headers={"X-Tenant-Id": tenant},
    )


# The triple-quoted block under a step is passed separately as `docstring`,
# not as part of the step name.
@given(parsers.parse('the assistant has indexed a document "{name}" containing:'))
def indexed_document(client, name, docstring):
    assert _upload(client, name, docstring).status_code == 200


@given(parsers.parse('visitor "{tenant}" has uploaded "{name}" containing:'))
def visitor_uploaded(client, tenant, name, docstring):
    assert _upload(client, name, docstring, tenant=tenant).status_code == 200


@given("the assistant has no documents indexed")
def no_documents(client):
    """Assert the precondition rather than assuming it.

    Each test gets its own store, so this is empty unless a step filled it --
    stating it explicitly keeps the scenario honest if that ever changes.
    """
    assert client.get("/api/documents").json()["documents"] == []


def apiheaders():
    return {"X-Tenant-Id": "tester"}


@when(parsers.parse('I ask "{question}"'))
def i_ask(client, context, question):
    context["response"] = client.post(
        "/api/query", json={"question": question}, headers=apiheaders()
    )


@when(parsers.parse('visitor "{tenant}" asks "{question}"'))
def visitor_asks(client, context, tenant, question):
    context["response"] = client.post(
        "/api/query", json={"question": question}, headers={"X-Tenant-Id": tenant}
    )


@when(parsers.parse('I ask "{question}" with history:'))
def i_ask_with_history(client, context, question, datatable):
    header, *rows = datatable
    history = [dict(zip(header, row)) for row in rows]
    context["response"] = client.post(
        "/api/query",
        json={"question": question, "history": history},
        headers=apiheaders(),
    )


@when(parsers.parse('visitor "{tenant}" lists documents'))
def visitor_lists(client, context, tenant):
    context["response"] = client.get(
        "/api/documents", headers={"X-Tenant-Id": tenant}
    )


@then(parsers.parse('the answer cites "{name}"'))
def answer_cites(context, name):
    sources = context["response"].json()["sources"]
    assert any(s["source"] == name for s in sources), sources


@then("no guardrail reports missing context")
def no_missing_context(context):
    guardrails = context["response"].json()["guardrails"]
    assert not any(g.startswith("grounding:no_documents") for g in guardrails)


@then("the assistant refuses to answer")
def refuses(context):
    guardrails = context["response"].json()["guardrails"]
    assert any(g.startswith("grounding") for g in guardrails)


@then("no source is cited")
def no_source(context):
    assert context["response"].json()["sources"] == []


@then("the request is rejected")
def rejected(context):
    assert context["response"].status_code == 400


@then("the request is blocked by a guardrail")
def blocked(context):
    body = context["response"].json()
    assert body["blocked"] is True
    assert any(g.startswith("input:") for g in body["guardrails"])


@then("the suspicious passage is excluded from the context")
def excluded(context):
    guardrails = context["response"].json()["guardrails"]
    assert any(g.startswith("context:") for g in guardrails), guardrails


@then(parsers.parse('"{name}" is not listed'))
def not_listed(context, name):
    names = [d["filename"] for d in context["response"].json()["documents"]]
    assert name not in names


@then("the question was not rewritten")
def not_rewritten(context):
    assert context["response"].json()["standalone_question"] is None


@then("the request is rejected as invalid")
def rejected_invalid(context):
    assert context["response"].status_code == 422


@then(parsers.parse('no source from "{name}" is cited'))
def no_source_from(context, name):
    sources = context["response"].json()["sources"]
    assert all(s["source"] != name for s in sources)
