//! `POST /v1/console/signup|login` (PUBLIC) + `GET /v1/console/me` /
//! `POST /v1/console/logout` (SESSION-authed) — the Community Edition Console
//! email+password auth surface.
//!
//! Contract:
//!   * `POST /v1/console/signup` → `{email, password}`. 201 + a session object
//!     (auto-login). 400 on shape failures; 409 on a duplicate email.
//!     **Open signup is intentional for a self-hosted CE** — the first (and
//!     usually only) operator bootstraps their own account; invite/approval
//!     gating is an Enterprise concern, not built here.
//!   * `POST /v1/console/login`  → `{email, password}`. 200 + a session object.
//!     EVERY failure (unknown email, wrong password, malformed email) is the
//!     SAME generic 401 after the SAME fixed delay — no enumeration oracle.
//!   * `GET  /v1/console/me`     → `{object:"console.account", email, created_at}`.
//!   * `POST /v1/console/logout` → bumps the account's session version
//!     (persisted), revoking EVERY outstanding token for the account; returns
//!     `{object:"console.session", deleted:true}`.
//!
//! The session object: `{object:"console.session", email, created_at, token,
//! token_type:"Bearer", expires_in}` — the SPA sends `token` back as
//! `Authorization: Bearer <token>` on every API call (see console_auth.rs).
//!
//! Secret handling: the password is argon2id-hashed on a blocking worker and
//! NEVER stored/logged/echoed; the request struct has no `Debug` derive; the
//! session token is returned to the caller once and never logged.

use crate::api_error::{error_response, internal_error, OpenAiJson};
use crate::auth::SharedAuthState;
use crate::console_accounts::{
    hash_password, normalize_email, validate_password, verify_password, CreateError,
};
use crate::console_auth::{ConsoleSession, SharedConsoleAuth, SESSION_TTL_SECS};
use axum::extract::ConnectInfo;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use routeplane_limits::auth_failures::AuthFailureTracker;
use routeplane_limits::now_unix_ms;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

/// Fixed delay applied to EVERY failed login before the generic 401 — blunts
/// online brute force and (with the dummy-verify below) flattens the
/// known-vs-unknown-email timing difference. Layered UNDER the per-source
/// throttle below.
const LOGIN_FAILURE_DELAY: Duration = Duration::from_millis(300);

/// Per-source-IP throttle for the PUBLIC console credential routes
/// (`/v1/console/signup` + `/v1/console/login`). Always-on for CE — these are
/// the only unauthenticated password endpoints, so they need brute-force
/// protection the keyed auth-failure tracker (which sits behind auth) can't
/// give. Shares one budget across signup + login so an attacker can't dodge the
/// login limit by pounding signup. Injected as a request extension.
pub type SharedConsoleThrottle = Arc<AuthFailureTracker>;

/// The IP a throttle decision is keyed on: the TCP peer address (`ConnectInfo`),
/// which is NOT client-spoofable (unlike `X-Forwarded-For`). A CE instance
/// behind a reverse proxy sees the proxy IP — over-throttling (the fail-safe
/// direction) rather than under-throttling; a proxied deployment should forward
/// the real peer or terminate closer to the client.
fn throttle_key(addr: &SocketAddr) -> String {
    addr.ip().to_string()
}

/// 429 with `Retry-After` when the source IP is over the credential-route
/// threshold. The body never reveals account state — pure rate-limit signal.
fn too_many_attempts(retry_after_secs: u64) -> Response {
    let mut resp = error_response(
        StatusCode::TOO_MANY_REQUESTS,
        "too_many_attempts",
        "Too many authentication attempts. Try again later.",
        "invalid_request_error",
        None,
    );
    if let Ok(v) = axum::http::HeaderValue::from_str(&retry_after_secs.to_string()) {
        resp.headers_mut().insert("retry-after", v);
    }
    resp
}

/// Signup/login request body. NO `Debug` derive — carries a plaintext password.
#[derive(serde::Deserialize)]
pub struct CredentialsRequest {
    pub email: String,
    pub password: String,
}

/// The one generic auth-failure envelope for login: never says WHICH of
/// email/password was wrong.
fn invalid_credentials() -> Response {
    error_response(
        StatusCode::UNAUTHORIZED,
        "invalid_credentials",
        "Invalid email or password.",
        "invalid_request_error",
        None,
    )
}

/// 401 for a session-only endpoint reached without a console session (e.g.
/// authed with an rp_ key, whose requests carry no `ConsoleSession`).
fn console_session_required() -> Response {
    error_response(
        StatusCode::UNAUTHORIZED,
        "console_session_required",
        "This endpoint requires a console session. Log in via POST /v1/console/login \
         and send the token as 'Authorization: Bearer <token>'.",
        "invalid_request_error",
        None,
    )
}

/// The session object returned by signup (201) and login (200).
fn session_response(
    status: StatusCode,
    email: &str,
    created_at: &Option<String>,
    token: String,
) -> Response {
    (
        status,
        Json(serde_json::json!({
            "object": "console.session",
            "email": email,
            "created_at": created_at,
            "token": token,
            "token_type": "Bearer",
            "expires_in": SESSION_TTL_SECS,
        })),
    )
        .into_response()
}

/// A real argon2id hash of a fixed non-secret placeholder, verified against on
/// login for an UNKNOWN email so the unknown-email and wrong-password paths
/// cost the same (no user-enumeration timing oracle). Computed once.
fn dummy_hash() -> &'static str {
    static DUMMY: OnceLock<String> = OnceLock::new();
    DUMMY
        .get_or_init(|| hash_password("routeplane-console-dummy-password").unwrap_or_default())
        .as_str()
}

/// `POST /v1/console/signup` (PUBLIC) — create an account + auto-login.
pub async fn signup(
    Extension(bridge): Extension<SharedConsoleAuth>,
    Extension(throttle): Extension<SharedConsoleThrottle>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    OpenAiJson(req): OpenAiJson<CredentialsRequest>,
) -> Response {
    // Per-IP throttle FIRST (before any argon2 work) — bounds signup-flood
    // argon2 DoS and shares the login budget. Each attempt counts toward the IP.
    let source = throttle_key(&peer);
    let now = now_unix_ms();
    if let routeplane_limits::auth_failures::AuthThrottle::Throttled { retry_after_secs } =
        throttle.check(&source, now)
    {
        return too_many_attempts(retry_after_secs);
    }
    throttle.record_failure(&source, now);
    let email = match normalize_email(&req.email) {
        Ok(e) => e,
        Err(msg) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_email",
                msg,
                "invalid_request_error",
                Some("email"),
            )
        }
    };
    if let Err(msg) = validate_password(&req.password) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_password",
            msg,
            "invalid_request_error",
            Some("password"),
        );
    }
    // argon2id is deliberately CPU-hard (~tens of ms) — hash on the blocking
    // pool so a signup never stalls a runtime worker. The password String moves
    // into the closure and drops there (no zeroization — `zeroize` would be a
    // new crate; accepted + documented residual for a self-host login).
    let password = req.password;
    let password_hash = match tokio::task::spawn_blocking(move || hash_password(&password)).await {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            tracing::error!("console signup: password hashing failed: {e}");
            return internal_error();
        }
        Err(e) => {
            tracing::error!("console signup: hashing task failed: {e}");
            return internal_error();
        }
    };
    // Spawn so the persist→swap pair completes even if the caller disconnects
    // (the providers_api cancellation-safety posture).
    let store = bridge.accounts.clone();
    let create_email = email.clone();
    match tokio::spawn(async move { store.create(create_email, password_hash).await }).await {
        Ok(Ok(account)) => {
            // 409: signup necessarily reveals email existence (it must refuse
            // the duplicate) — acceptable for an open self-host signup; LOGIN
            // stays enumeration-safe.
            match bridge.issue(&account.email, account.session_version) {
                Ok(token) => {
                    tracing::info!("console account created"); // no email/PII in logs
                    session_response(
                        StatusCode::CREATED,
                        &account.email,
                        &account.created_at,
                        token,
                    )
                }
                Err(e) => {
                    tracing::error!("console signup: session issue failed: {e}");
                    internal_error()
                }
            }
        }
        Ok(Err(CreateError::Duplicate)) => error_response(
            StatusCode::CONFLICT,
            "email_already_registered",
            "An account with this email already exists. Log in instead.",
            "invalid_request_error",
            Some("email"),
        ),
        Ok(Err(CreateError::Store(e))) => {
            // `e` is a store/persist error string — never contains a hash.
            tracing::error!("console signup: account persist failed: {e}");
            internal_error()
        }
        Err(e) => {
            tracing::error!("console signup: create task failed: {e}");
            internal_error()
        }
    }
}

/// `POST /v1/console/login` (PUBLIC) — verify, mint a session.
pub async fn login(
    Extension(bridge): Extension<SharedConsoleAuth>,
    Extension(throttle): Extension<SharedConsoleThrottle>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    OpenAiJson(req): OpenAiJson<CredentialsRequest>,
) -> Response {
    // Per-IP throttle FIRST — bound online brute force before the (CPU-hard)
    // argon2 verify runs. Shares the signup budget for this source IP.
    let source = throttle_key(&peer);
    let now = now_unix_ms();
    if let routeplane_limits::auth_failures::AuthThrottle::Throttled { retry_after_secs } =
        throttle.check(&source, now)
    {
        return too_many_attempts(retry_after_secs);
    }
    // A malformed email cannot match an account; it still walks the full
    // dummy-verify + delay path below so the response is indistinguishable.
    let email = normalize_email(&req.email).unwrap_or_default();
    let account = bridge.accounts.find(&email);
    // ALWAYS run one argon2 verify — against the real hash when the account
    // exists, against the fixed dummy hash otherwise — so unknown-email and
    // wrong-password cost the same. On the blocking pool (CPU-hard).
    let phc = account
        .as_ref()
        .map(|a| a.password_hash.clone())
        .unwrap_or_else(|| dummy_hash().to_string());
    let password = req.password;
    let verified = tokio::task::spawn_blocking(move || verify_password(&password, &phc))
        .await
        .unwrap_or(false);
    match account {
        Some(account) if verified => match bridge.issue(&account.email, account.session_version) {
            Ok(token) => {
                session_response(StatusCode::OK, &account.email, &account.created_at, token)
            }
            Err(e) => {
                tracing::error!("console login: session issue failed: {e}");
                internal_error()
            }
        },
        // Generic failure: fixed delay + one envelope for every cause. No
        // email, no cause, nothing logged above debug (no credential-stuffing
        // amplification via logs).
        _ => {
            // Record the failure against the source IP so repeated wrong
            // credentials trip the throttle above; then the fixed delay + one
            // generic envelope (enumeration-safe).
            throttle.record_failure(&source, now);
            tokio::time::sleep(LOGIN_FAILURE_DELAY).await;
            tracing::debug!("console login rejected");
            invalid_credentials()
        }
    }
}

/// `GET /v1/console/me` (SESSION-authed) — the session's own account identity.
/// The `ConsoleSession` extension exists ONLY when `auth_middleware` accepted a
/// console session; an rp_-key-authed request gets the 401 below.
pub async fn me(session: Option<Extension<ConsoleSession>>) -> Response {
    let Some(Extension(session)) = session else {
        return console_session_required();
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "object": "console.account",
            "email": session.email,
            "created_at": session.created_at,
        })),
    )
        .into_response()
}

/// `GET /v1/console/api-key` (SESSION-authed) — the gateway `rp_` key this
/// console session authorizes as, so the operator can copy it into an SDK/app.
///
/// INTENTIONAL REVEAL: this returns the operator's OWN gateway key to an
/// authenticated console session — that is the endpoint's purpose. It is
/// session-only: an rp_-key-authed request carries no `ConsoleSession`
/// extension and gets the same 401 as `/me`; an unauthenticated request never
/// clears `auth_middleware`. The key value is NEVER logged.
pub async fn api_key(
    Extension(bridge): Extension<SharedConsoleAuth>,
    Extension(auth): Extension<SharedAuthState>,
    session: Option<Extension<ConsoleSession>>,
) -> Response {
    let Some(Extension(_session)) = session else {
        return console_session_required();
    };
    // Lock-free registry snapshot (`ArcSwap::load`). `console_key` is
    // server-side config (see console_auth.rs), not a request-supplied secret,
    // and we are post-auth — a plain map probe needs no constant-time
    // treatment. Boot validates the key is registered, but the registry is
    // hot-swappable, so fall back to the key value as the display name if the
    // entry has vanished (the key itself is already being revealed by design).
    let name = auth
        .load()
        .keys
        .get(&bridge.console_key)
        .map(|vk| vk.name.clone())
        .unwrap_or_else(|| bridge.console_key.clone());
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "object": "console.api_key",
            "name": name,
            "key": bridge.console_key,
        })),
    )
        .into_response()
}

/// `POST /v1/console/logout` (SESSION-authed) — REAL revocation: bump the
/// account's persisted session version so every outstanding token (this one
/// included) is dead on the next request. The client drops its copy.
pub async fn logout(
    Extension(bridge): Extension<SharedConsoleAuth>,
    session: Option<Extension<ConsoleSession>>,
) -> Response {
    let Some(Extension(session)) = session else {
        return console_session_required();
    };
    // Spawned for the same cancellation-safety reason as providers_api: the
    // persist→swap pair must complete even if the caller disconnects.
    let store = bridge.accounts.clone();
    let email = session.email;
    match tokio::spawn(async move { store.bump_session_version(&email).await }).await {
        // `Ok(None)` (account deleted underneath) is still a successful logout.
        Ok(Ok(_)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "object": "console.session",
                "deleted": true,
            })),
        )
            .into_response(),
        Ok(Err(e)) => {
            tracing::error!("console logout: revocation persist failed: {e}");
            internal_error()
        }
        Err(e) => {
            tracing::error!("console logout: task failed: {e}");
            internal_error()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{auth_middleware, shared_auth_state, AuthState, VirtualKey};
    use crate::console_accounts::ConsoleAccountStore;
    use crate::console_auth::ConsoleAuthBridge;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use axum::routing::{get, post};
    use axum::Router;
    use std::sync::Arc;
    use tower::ServiceExt;

    /// A router wired exactly like main.rs: public signup/login + an authed
    /// section (auth_middleware + SharedAuthState + SharedConsoleAuth) carrying
    /// me/logout AND a data-plane probe that echoes the resolved VirtualKey —
    /// proving a session authorizes a normal data-plane call as the console
    /// tenant without the browser ever seeing the rp_ key.
    fn test_stack() -> (Router, SharedConsoleAuth) {
        let auth = AuthState::load_from_json(
            r#"{"keys":[
                {"name":"Console Key","routeplane_key":"rp_console_test","provider_keys":{},"tenant_id":"t_console"},
                {"name":"Other Key","routeplane_key":"rp_other","provider_keys":{},"tenant_id":"t_other"}
            ]}"#,
            "test",
        )
        .expect("registry");
        let auth = shared_auth_state(auth);
        let bridge: SharedConsoleAuth = Arc::new(ConsoleAuthBridge::new(
            b"unit-test-session-secret-0123456789",
            "rp_console_test".into(),
            Arc::new(ConsoleAccountStore::ephemeral()),
        ));
        let authed = Router::new()
            .route("/v1/console/me", get(me))
            .route("/v1/console/api-key", get(api_key))
            .route("/v1/console/logout", post(logout))
            .route(
                "/probe",
                get(
                    |axum::Extension(vk): axum::Extension<VirtualKey>| async move {
                        vk.resolved_tenant_id()
                    },
                ),
            )
            .layer(axum::middleware::from_fn(auth_middleware))
            .layer(axum::Extension(auth))
            .layer(axum::Extension(bridge.clone()));
        // A lenient throttle (very high threshold) so the functional tests below
        // are unaffected by the credential-route rate limit; a dedicated test
        // exercises the throttle with a tight config. MockConnectInfo supplies
        // the ConnectInfo<SocketAddr> the handlers now extract.
        let throttle: SharedConsoleThrottle = Arc::new(AuthFailureTracker::new(
            routeplane_limits::auth_failures::AuthFailureConfig {
                threshold: 100_000,
                ..Default::default()
            },
        ));
        let app = Router::new()
            .route("/v1/console/signup", post(signup))
            .route("/v1/console/login", post(login))
            .layer(axum::Extension(bridge.clone()))
            .layer(axum::Extension(throttle))
            .layer(axum::extract::connect_info::MockConnectInfo(
                "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            ))
            .merge(authed);
        (app, bridge)
    }

    /// A single-route login app whose throttle trips after `threshold` failures,
    /// keyed on a fixed mock peer IP — for the throttle test only.
    fn throttled_login_app(threshold: u64) -> Router {
        let bridge: SharedConsoleAuth = Arc::new(ConsoleAuthBridge::new(
            b"unit-test-session-secret-0123456789",
            "rp_console_test".into(),
            Arc::new(ConsoleAccountStore::ephemeral()),
        ));
        let throttle: SharedConsoleThrottle = Arc::new(AuthFailureTracker::new(
            routeplane_limits::auth_failures::AuthFailureConfig {
                threshold,
                window_ms: 300_000,
                backoff_base_ms: 2_000,
                backoff_cap_ms: 900_000,
                slots: 64,
            },
        ));
        Router::new()
            .route("/v1/console/login", post(login))
            .layer(axum::Extension(bridge))
            .layer(axum::Extension(throttle))
            .layer(axum::extract::connect_info::MockConnectInfo(
                "203.0.113.7:0".parse::<SocketAddr>().unwrap(),
            ))
    }

    fn json_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn bearer_get(uri: &str, token: &str) -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri(uri)
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn signup_token(app: &Router) -> String {
        let resp = app
            .clone()
            .oneshot(json_post(
                "/v1/console/signup",
                serde_json::json!({"email":"Op@Example.com","password":"a-long-password"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "console.session");
        assert_eq!(v["email"], "op@example.com", "email is lowercased");
        assert_eq!(v["token_type"], "Bearer");
        assert_eq!(v["expires_in"], SESSION_TTL_SECS);
        v["token"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn signup_autologin_me_and_dataplane_probe_via_session() {
        let (app, _) = test_stack();
        let token = signup_token(&app).await;

        // The session token authorizes /me…
        let resp = app
            .clone()
            .oneshot(bearer_get("/v1/console/me", &token))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "console.account");
        assert_eq!(v["email"], "op@example.com");

        // …and a NORMAL data-plane call, resolving to the console tenant —
        // the browser never presented (or received) the rp_ key.
        let resp = app
            .clone()
            .oneshot(bearer_get("/probe", &token))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        assert_eq!(&body[..], b"t_console");

        // A signup response must never leak a hash.
        let resp = app
            .clone()
            .oneshot(json_post(
                "/v1/console/signup",
                serde_json::json!({"email":"second@example.com","password":"a-long-password"}),
            ))
            .await
            .unwrap();
        let v = body_json(resp).await;
        assert!(
            !v.to_string().contains("argon2"),
            "no hash material in any response"
        );
    }

    #[tokio::test]
    async fn signup_rejects_bad_email_short_password_and_duplicates() {
        let (app, _) = test_stack();
        // Bad email shape → 400.
        let resp = app
            .clone()
            .oneshot(json_post(
                "/v1/console/signup",
                serde_json::json!({"email":"not-an-email","password":"a-long-password"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(resp).await["error"]["code"], "invalid_email");

        // Password < 10 chars → 400 (and the password is not echoed).
        let resp = app
            .clone()
            .oneshot(json_post(
                "/v1/console/signup",
                serde_json::json!({"email":"op@example.com","password":"short-pw!"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["code"], "invalid_password");
        assert!(!v.to_string().contains("short-pw!"));

        // Duplicate email (case-insensitively) → 409.
        let _ = signup_token(&app).await;
        let resp = app
            .clone()
            .oneshot(json_post(
                "/v1/console/signup",
                serde_json::json!({"email":"OP@example.COM","password":"another-long-pw"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(
            body_json(resp).await["error"]["code"],
            "email_already_registered"
        );
    }

    #[tokio::test]
    async fn login_happy_path_and_generic_401_on_any_failure() {
        let (app, _) = test_stack();
        let _ = signup_token(&app).await;

        // Correct credentials → 200 + a working session.
        let resp = app
            .clone()
            .oneshot(json_post(
                "/v1/console/login",
                serde_json::json!({"email":"op@example.com","password":"a-long-password"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let token = body_json(resp).await["token"].as_str().unwrap().to_string();
        let resp = app
            .clone()
            .oneshot(bearer_get("/v1/console/me", &token))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Wrong password and unknown email: the SAME generic envelope.
        let wrong_pw = app
            .clone()
            .oneshot(json_post(
                "/v1/console/login",
                serde_json::json!({"email":"op@example.com","password":"wrong-password"}),
            ))
            .await
            .unwrap();
        let unknown = app
            .clone()
            .oneshot(json_post(
                "/v1/console/login",
                serde_json::json!({"email":"ghost@example.com","password":"a-long-password"}),
            ))
            .await
            .unwrap();
        assert_eq!(wrong_pw.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(unknown.status(), StatusCode::UNAUTHORIZED);
        let (a, b) = (body_json(wrong_pw).await, body_json(unknown).await);
        assert_eq!(a, b, "no oracle: identical body for both failure causes");
        assert_eq!(a["error"]["code"], "invalid_credentials");
    }

    #[tokio::test]
    async fn api_key_returns_console_key_only_with_a_session() {
        let (app, _) = test_stack();
        let token = signup_token(&app).await;

        // A valid console session → 200 with the bridged rp_ key + its
        // registry display name (the intentional own-key reveal).
        let resp = app
            .clone()
            .oneshot(bearer_get("/v1/console/api-key", &token))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "console.api_key");
        assert_eq!(v["name"], "Console Key");
        assert_eq!(v["key"], "rp_console_test");

        // No credentials at all → auth_middleware 401 (never reaches the
        // handler, no key in the body).
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/console/api-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(
            !body_json(resp)
                .await
                .to_string()
                .contains("rp_console_test"),
            "no key material for unauthenticated callers"
        );

        // An rp_-key-authed request (valid gateway auth, NOT a console
        // session) → the same console_session_required 401 as /me.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/console/api-key")
                    .header("x-routeplane-api-key", "rp_other")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["code"], "console_session_required");
        assert!(
            !v.to_string().contains("rp_console_test"),
            "no key material for rp_-key callers"
        );
    }

    #[tokio::test]
    async fn bad_expired_and_revoked_tokens_are_401_and_rp_key_path_unaffected() {
        let (app, bridge) = test_stack();
        let token = signup_token(&app).await;

        // Garbage / unsigned tokens → the standard 401.
        for bad in ["garbage", "eyJhbGciOiJub25lIn0.e30."] {
            let resp = app
                .clone()
                .oneshot(bearer_get("/probe", bad))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::UNAUTHORIZED, "token={bad}");
        }

        // An EXPIRED token signed with the right secret → 401.
        let now = chrono::Utc::now().timestamp().max(0) as u64;
        let expired = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &serde_json::json!({"sub":"op@example.com","iat":now-7200,"exp":now-3600,"sv":0}),
            &jsonwebtoken::EncodingKey::from_secret(b"unit-test-session-secret-0123456789"),
        )
        .unwrap();
        let resp = app
            .clone()
            .oneshot(bearer_get("/probe", &expired))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Logout revokes: the old token dies on the very next request.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/console/logout")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["deleted"], true);
        let resp = app
            .clone()
            .oneshot(bearer_get("/v1/console/me", &token))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // The rp_-key path is untouched: both headers still authenticate, and
        // an rp_ key on a session-only endpoint gets the session-required 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("x-routeplane-api-key", "rp_other")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        assert_eq!(&body[..], b"t_other", "rp_ key resolves its OWN tenant");
        let resp = app
            .clone()
            .oneshot(bearer_get("/probe", "rp_console_test"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "Bearer rp_ fallback intact");
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/console/me")
                    .header("x-routeplane-api-key", "rp_other")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            body_json(resp).await["error"]["code"],
            "console_session_required"
        );

        let _ = bridge; // keep the bridge alive through the test
    }

    #[tokio::test]
    async fn login_throttles_after_threshold_failures_from_one_ip() {
        // threshold=3: the 4th wrong-credential attempt from the same IP must be
        // refused with 429 + Retry-After, before any account exists (pure
        // brute-force bound). Proves the throttle actually throttles.
        let app = throttled_login_app(3);
        let attempt = || {
            app.clone().oneshot(json_post(
                "/v1/console/login",
                serde_json::json!({ "email": "nobody@example.com", "password": "wrong-password-xyz" }),
            ))
        };
        // First 3 are allowed through (and fail 401 invalid_credentials).
        for i in 0..3 {
            let resp = attempt().await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "attempt {i} should be a normal 401, not throttled yet"
            );
        }
        // The 4th is throttled.
        let resp = attempt().await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "over-threshold attempt must be 429"
        );
        assert!(
            resp.headers().get("retry-after").is_some(),
            "429 must carry a Retry-After"
        );
        assert_eq!(body_json(resp).await["error"]["code"], "too_many_attempts");
    }
}
