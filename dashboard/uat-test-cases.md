# UAT: CE setup and retained trace inspection

These cases define acceptance for the bundled Community Edition Console. They
are a test plan, not a claim of successful execution. The environment owner
must approve the browser/fixture window before execution. No cloud target,
Rust compilation or container lifecycle action is part of this frontend plan.

Prerequisites: exact reviewed frontend build, owner-provided pinned public CE
gateway image/config, synthetic `self_hosted` fixture with request receipts,
empty model catalogue, scoped Console account, and the optional retained-log request_id field.
Credentials must flow from the test environment's secret configuration through the existing
scripted login/session setup; never type or print them interactively. Keep
session-bearing captures private and publish only checked derivatives.

| Case | Priority | Steps and required evidence |
|---|---|---|
| TC-01 Secure defaults | Critical | With signup off, fresh unauthenticated signup is403 and logs401. Retain raw status/body receipts. No provider request is made. |
| TC-02 Controlled setup | Critical | Owner enables signup in the restricted synthetic profile; create scoped account via existing auth script, disable signup and sign in again. Verify configured tenant scope and no anonymous logs access. |
| TC-03 Empty catalogue | Critical | Open Playground with `data:[]`; Run disabled initially and with only a model. Enter exact fixture model and `self_hosted`; Run enables. No request before click. |
| TC-04 Streaming | Critical | Click Run once; verify incremental actual SSE output and successful DONE, exact selected model/provider header in fixture receipt. No alternate dispatch. Compare visible/copied Gateway request ID to the actual response header; ID remains available on a later stream error and resets on the next run. Missing headers never invent an ID. |
| TC-05 Buffered control | Critical | Existing API/SDK sends identical model/provider/messages with `stream:false`; inspect successful response and fixture receipt. No buffered UI toggle implied. |
| TC-06 Errors/cancellation | High | Missing credential, unavailable upstream and midstream disconnect show actionable errors without a hidden retry or success indication. Stop interrupts one stream; run a fresh explicit recovery call. Retain partial output honestly. |
| TC-07 Trace inspection | Critical | Search the actual response req_ ID; inspect the matching existing log_ row using Enter and Space, compare displayed/copied request_id byte-for-byte. Multiple attempts remain separate rows. Missing historical IDs show Not recorded; nonmatching search shows empty state. |
| TC-08 Mobile initial layout | High | At390×844, sign-in, Playground and populated Logs have document scrollWidth≤clientWidth; no overlapped controls. Capture screenshots/rectangles. Scroll wide table inside its bounded region, not page. |
| TC-09 Desktop and resize | High | At1280×800 preserve sidebar collapse and row drawer behavior. Resize390→1280→390 with navigation open/closed. Assert no page overflow/obscured controls. |
| TC-10 Keyboard focus | High | Use Tab/Enter to open navigation, activate Logs, search, open row, copy request ID. Escape/outside-close restores focus to opener; resized desktop closes mobile modal and focuses visible desktop nav. Check visible focus screenshots, no trap after close. |
| TC-11 Reload/history | High | Reload authenticated page; server-returned rows remain usable while still retained. After owner-controlled ring reset/eviction, absence is honest, with no reconstructed correlation. |
| TC-12 Clipboard denial | High | Deny clipboard access; Copy reports failure rather than success, and displayed ID remains manually selectable. Restore permission and verify actual clipboard value. |

Record exact source/image/config hashes, timestamp, viewport, per-case verdict,
console errors, screenshots and sanitized fixture receipts in the lab evidence
directory. Execute against the exact published CE image digest for delivery
acceptance. Source checks or tests of another image do not prove that delivery.
