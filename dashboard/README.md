# Community Edition Console: controlled self-hosted setup

This bundled UI talks to the CE gateway, not a separate Control Plane. Keep it
on a trusted local/private origin during setup. Use TLS before remote access.
The following are operator settings, not browser fields:

1. Configure the existing `self_hosted` provider and its model in your upstream.
   `SELF_HOSTED_BASE_URL` is the server root **without `/v1`**; the adapter adds
   the API path. Configure its credential in the selected gateway key's provider
   map as required by that upstream. Do not paste provider secrets into chat or
   commit them to source control.
2. The bundled CE image sets `RP_CONSOLE_DIR` to its built Console assets;
   outside that image, set it to the directory containing those assets to serve
   the UI from the gateway origin. Bind
   `RP_CONSOLE_KEY` explicitly to the intended registered gateway key. All CE
   Console accounts authorize as this one configured key: this is **not** an
   account-per-tenant provisioning mechanism. An ambiguous multi-key registry
   without explicit binding refuses startup when signup is enabled or existing
   accounts can authorize. A gateway with no accounts and signup disabled may
   start without that explicit binding, but this does not authorize Console use.
3. Keep `RP_CONSOLE_SIGNUP` unset/off normally. Signup returns **403** by default;
   anonymous `/v1/logs` remains **401**. For initial account creation, an operator
   may set `RP_CONSOLE_SIGNUP=on` only during a controlled, access-restricted
   setup window. Anyone who can register during that window obtains the bound
   key's Console privileges. Create the operator account, then turn signup off
   and restart; existing accounts can still sign in.
4. Persist the configured account file (`RP_CONSOLE_ACCOUNTS_FILE`, default
   `configs/console-accounts.json`) with restricted permissions. Supply a strong
   stable `RP_CONSOLE_SESSION_SECRET` through your secret mechanism if sessions
   must survive restarts; otherwise the gateway generates a per-boot secret.
   The browser stores its signed session locally. Do not publish session tokens
   or browser storage snapshots. Set `RP_TRUSTED_PROXY_HOPS` only to the actual
   trusted proxy count if a reverse proxy fronts the Console.

## A self-hosted model with an empty catalogue

Open **Playground**, enter the exact upstream model (for example a model you
have already installed in Ollama), and explicitly enter **`self_hosted`** in
Provider. Run stays disabled until both fields are valid. The catalogue and
provider suggestions are not credential or model-health checks. This screen
does not discover models, accept upstream URLs, or select a fallback provider.
Click **Run** for streaming; **Stop** aborts that request. A truncated stream is
reported as an error while retaining the partial text.

For a buffered API control, send the same model and messages to the same gateway
`POST /v1/chat/completions` with `stream: false`, your gateway authorization and
`x-routeplane-provider: self_hosted`. For example, the non-secret body shape is:

```json
{"model":"YOUR_INSTALLED_MODEL","messages":[{"role":"user","content":"Say hello."}],"stream":false}
```

Use a securely supplied gateway key in `x-routeplane-api-key`; do not hardcode a
real key into scripts, shell history or examples. A successful buffered control
does not prove a successful stream: inspect both responses and their upstream
receipts. Missing credentials or an unavailable upstream remain errors; check
the configured key's provider map and the operator-owned endpoint.

## Inspect a retained request

Copy **Gateway request ID** beside the Playground response and search for it
in **Logs & Traces**. The UI reads the actual `x-routeplane-request-id` header,
falling back to its `x-routeplane-trace-id` alias (`req_…`); it never uses the
provider completion ID or a W3C/OpenTelemetry trace ID. The ID is available
once response headers arrive, including failed/partial streams, and resets
before each new request. If neither header is present, no ID is invented.
The existing `log_…` row identifier remains separate. Open a row by click,
Enter or Space; the detail drawer shows and copies `request_id` when recorded.
Older rows show **Not recorded**, never a fabricated correlation. The search
only covers rows returned by the bounded in-memory history: evicted/restarted
history cannot be recovered here. Escape closes a drawer and restores focus.

## Frontend-only checks

`npm ci --ignore-scripts`, `npm test`, `npm run typecheck` and `npm run build`
are frontend checks; none starts or builds the Rust gateway. `build` runs the
focused tests before producing the bundle. Browser acceptance is documented in
[uat-test-cases.md](uat-test-cases.md); passing unit tests alone is not delivered
image or real-provider acceptance.

The collapsed desktop rail is 64px wide. Its header contains only a labelled 32px
Expand button, centered inside that rail; the full brand lockup returns when
expanded. Keeping a logo plus toggle in the 64px header previously pushed the
toggle beneath the adjacent main header. The sizing regression test does not
replace the browser collapse→expand/resize acceptance case.
