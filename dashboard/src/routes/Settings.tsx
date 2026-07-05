import { useQuery } from "@tanstack/react-query";
import { ExternalLink, LogOut, Mail, Moon, Sun } from "lucide-react";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/layout/PageHeader";
import { EnterpriseHint } from "@/components/EnterpriseHint";
import { KeyValueList } from "@/components/ui/misc";
import { Switch } from "@/components/ui/switch";
import { useTheme } from "@/lib/theme";
import { fetchMe, signOut } from "@/lib/auth";
import { formatDateTime } from "@/lib/utils";

export function Settings() {
  const { theme, toggle } = useTheme();
  const me = useQuery({ queryKey: ["console-me"], queryFn: fetchMe, staleTime: 300_000 });

  return (
    <>
      <PageHeader title="Settings" description="Console preferences and your account. Theme is stored locally; your account lives on the gateway." />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Appearance" description="Only the theme preference is persisted (locally)." />
          <CardBody>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-sm">
                {theme === "dark" ? <Moon size={15} /> : <Sun size={15} />}
                Dark mode
              </span>
              <Switch checked={theme === "dark"} onCheckedChange={toggle} />
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Account" description="The email account this console session belongs to." />
          <CardBody className="space-y-4">
            <KeyValueList
              items={[
                {
                  label: "Signed in as",
                  value: (
                    <span className="inline-flex items-center gap-2 text-xs">
                      <Mail size={13} className="text-muted-foreground" />
                      {me.data?.email ?? "…"}
                    </span>
                  ),
                },
                ...(me.data?.created_at ? [{ label: "Member since", value: formatDateTime(me.data.created_at) }] : []),
              ]}
            />
            <Button variant="outline" onClick={() => void signOut().then(() => window.location.reload())}>
              <LogOut size={15} /> Sign out
            </Button>
          </CardBody>
        </Card>
      </div>

      <div className="mt-4">
        <EnterpriseHint title="Two-factor authentication & SSO">
          TOTP two-factor, SSO/SCIM (SAML &amp; OIDC), members &amp; RBAC, and entitlement management are Routeplane
          Enterprise capabilities. The Community Edition is single-tenant with email + password accounts.
        </EnterpriseHint>
      </div>

      <Card className="mt-4">
        <CardHeader title="About" action={<Badge tone="primary">Community Edition</Badge>} />
        <CardBody className="space-y-3 text-sm text-muted-foreground">
          <p>
            The Routeplane CE Console is a self-hosted dashboard for the Community Edition gateway. It talks only to the
            local data plane; your session token is the only thing stored in this browser.
          </p>
          <KeyValueList
            items={[
              { label: "Edition", value: "Community Edition" },
              { label: "License", value: "Apache-2.0" },
              {
                label: "Website",
                value: (
                  <a
                    href="https://routeplane.ai"
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1 text-primary hover:underline"
                  >
                    routeplane.ai <ExternalLink size={12} />
                  </a>
                ),
              },
            ]}
          />
        </CardBody>
      </Card>
    </>
  );
}
