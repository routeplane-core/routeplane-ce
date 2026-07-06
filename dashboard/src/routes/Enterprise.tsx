// Enterprise-only feature pages for the Community Edition.
//
// Every export here renders the shared <EnterpriseUpsell/> and nothing else — no
// enterprise endpoints, no request/response types, no thresholds, no credentials.
// The copy below is drawn only from Routeplane's public positioning. This file is
// the complete enterprise footprint of the CE Console.
import { EnterpriseUpsell } from "@/components/EnterpriseUpsell";

export function SmartRouting() {
  return (
    <EnterpriseUpsell
      title="Smart Routing"
      summary="Configure the smart-routing algorithm pipeline and simulate how requests are steered across providers by cost, latency, and quality."
      capabilities={[
        "Ordered algorithm pipeline (cost / latency / quality objectives)",
        "Routing simulator to preview candidate ordering before rollout",
        "Lock-free quota tracking on the hot path",
      ]}
    />
  );
}

export function RoutingPolicies() {
  return (
    <EnterpriseUpsell
      title="Routing Policies"
      summary="Declarative, versioned routing configs — conditional routing, load balancing, and canary weights managed centrally."
      capabilities={[
        "Fallback, load-balance, and conditional routing configs",
        "Inline per-request routing via x-routeplane-config",
        "Central policy management with audit trail",
      ]}
    />
  );
}

export function Credits() {
  return (
    <EnterpriseUpsell
      title="Smart Credits"
      summary="A prepaid, multi-currency credits wallet alongside bring-your-own-key, with a durable balance and settlement ledger."
      capabilities={[
        "Prepaid wallet with top-up and rate card",
        "Durable transaction ledger and invoices",
        "Multi-currency FinOps down to the business process",
      ]}
    />
  );
}

export function Cost() {
  return (
    <EnterpriseUpsell
      title="Cost Attribution"
      summary="Attribute spend in any currency down to the model, key, use-case, and business process, with exportable FinOps reports."
      capabilities={[
        "Cost breakdown by model / key / use-case / process",
        "Multi-currency conversion and reporting",
        "FinOps export + durable telemetry retention",
      ]}
    />
  );
}

export function Guardrails() {
  return (
    <EnterpriseUpsell
      title="Advanced Guardrails"
      summary="Beyond CE's deterministic PII masking: configurable detectors, ML-based prompt-injection and moderation, tokenization, and an egress firewall."
      capabilities={[
        "Prompt-injection & jailbreak detection (deterministic + ML)",
        "Content moderation and reversible PII tokenization",
        "Webhook + off-path detector framework with a decision cache",
      ]}
    />
  );
}

export function Agents() {
  return (
    <EnterpriseUpsell
      title="Agentic Security"
      summary="The MCP gateway and agent governance: per-tool-call authorization, agent-run traces, human-in-the-loop, and integrated threat detection."
      capabilities={[
        "MCP tool-call authorization policies",
        "Agent-run traces and threat detections",
        "Human-in-the-loop approvals and signed receipts",
      ]}
    />
  );
}

export function Residency() {
  return (
    <EnterpriseUpsell
      title="Sovereignty & Residency"
      summary="Region-locked sovereign routing enforced per request, with a residency ledger of what was routed in-region, sovereign-routed, or blocked."
      capabilities={[
        "Per-request region-locked routing enforcement",
        "Residency ledger and summary (DPDP / India-first, global from day one)",
        "Data-residency classification wired to routing",
      ]}
    />
  );
}

export function Compliance() {
  return (
    <EnterpriseUpsell
      title="Compliance & Artifacts"
      summary="Compliance-framework posture with a signed, hash-chained audit ledger and exportable, verifiable audit artifacts for regulators."
      capabilities={[
        "Org compliance-framework default-deny gating",
        "Hash-chained tamper-evident audit ledger",
        "Generate and download verifiable audit artifacts",
      ]}
    />
  );
}

export function Prompts() {
  return (
    <EnterpriseUpsell
      title="Prompt Registry"
      summary="A versioned prompt registry with labels, experiments, and per-variant analytics — move labels between versions without touching client code."
      capabilities={[
        "Versioned prompts with movable labels",
        "A/B experiments with per-variant analytics",
        "Attribution of usage events to prompt versions",
      ]}
    />
  );
}

export function AuditLog() {
  return (
    <EnterpriseUpsell
      title="Admin Audit Log"
      summary="A tamper-evident, hash-chained record of every administrative action, with redacted before/after states for compliance review."
      capabilities={[
        "Hash-chained admin action records",
        "Redacted before/after state capture",
        "Exportable for audit and regulator review",
      ]}
    />
  );
}
