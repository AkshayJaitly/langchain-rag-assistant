Feature: Visitors are isolated from each other
  A visitor sees the shared sample documents and their own uploads, and nobody
  else's. This is isolation between visitors, not authentication.

  Scenario: One visitor cannot see another's upload
    Given visitor "alice" has uploaded "offer.txt" containing:
      """
      The offered salary is 200000 per year.
      """
    When visitor "bob" lists documents
    Then "offer.txt" is not listed

  Scenario: One visitor cannot retrieve another's content
    Given visitor "alice" has uploaded "offer.txt" containing:
      """
      The offered salary is 200000 per year.
      """
    When visitor "bob" asks "What is the offered salary?"
    Then no source from "offer.txt" is cited
