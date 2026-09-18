Feature: Grounded answers with citations
  The assistant answers only from the documents it has indexed, and says so
  when it cannot.

  Scenario: A question answered by the documents cites its source
    Given the assistant has indexed a document "policy.txt" containing:
      """
      Hybrid employees attend the office at least eight days per calendar month.
      The connectivity allowance is 60 dollars per month.
      """
    When I ask "How many office days per month are required?"
    Then the answer cites "policy.txt"
    And no guardrail reports missing context

  Scenario: With nothing indexed the assistant refuses rather than inventing
    Given the assistant has no documents indexed
    When I ask "What is the parental leave policy?"
    Then the assistant refuses to answer
    And no source is cited

  Scenario: An empty question is rejected outright
    When I ask "   "
    Then the request is rejected
