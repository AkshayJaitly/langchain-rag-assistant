# 004 — Per-visitor document isolation

## Goal

Stop every visitor from reading every other visitor's uploads. The index is
currently one global collection with no notion of who uploaded what.

## Constraints

- Inherits C-1, C-2.
- **This is isolation, not authentication.** There is no account system and no
  free way to add one here. A tenant id is a browser-held opaque identifier: it
  keeps visitors out of each other's documents by default, and it is not a
  security boundary against someone who forges the header. The README must say
  this plainly.

## Behaviour

- **AC-1** Every chunk carries a `tenant_id`. Uploads use the caller's tenant;
  the bundled samples use the reserved tenant `public`.
- **AC-2** Retrieval for tenant `T` searches `T` and `public`, and nothing else.
- **AC-3** `GET /api/documents` lists the caller's documents plus the public
  samples, and marks which are shared.
- **AC-4** A caller with no tenant header is treated as `public`-read-only:
  they see the samples and their uploads go to an anonymous tenant.
- **AC-5** A caller cannot delete or overwrite another tenant's document; a
  re-upload of the same filename replaces only their own copy.
- **AC-6** The tenant id is generated in the browser, stored in
  `localStorage`, and sent as `X-Tenant-Id`.

## Non-goals

Accounts, login, sharing between tenants, server-side authorisation.

## Verification

`tests/integration/test_tenancy.py`, `features/tenancy.feature`.
