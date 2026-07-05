import { ExternalLink, KeyRound, LogOut, Moon, Sun } from "lucide-react";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/layout/PageHeader";
import { KeyValueList } from "@/components/ui/misc";
import { Switch } from "@/components/ui/switch";
import { useTheme } from "@/lib/theme";
import { getStoredKey, signOut } from "@/lib/auth";

function maskKey(k: string | null): string {
  if (!k) return "—";
  return k.length <= 10 ? k : `${k.slice(0, 7)}…${k.slice(-4)}`;
}

export function Settings() {
  const { theme, toggle } = useTheme();
  const key = getStoredKey();

  return (
    <>
      <PageHeader title="Settings" description="Console preferences and your gateway connection. Everything here is local to this browser." />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Appearance" description="Only the theme preference is persisted." />
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
          <CardHeader title="Gateway connection" description="The rp_ key this Console authenticates with." />
          <CardBody className="space-y-4">
            <KeyValueList
              items={[
                {
                  label: "Connected key",
                  value: (
                    <span className="inline-flex items-center gap-2 font-mono text-xs">
                      <KeyRound size={13} className="text-muted-foreground" />
                      {maskKey(key)}
                    </span>
                  ),
                },
              ]}
            />
            <Button
              variant="outline"
              onClick={() => {
                signOut();
                window.location.reload();
              }}
            >
              <LogOut size={15} /> Sign out
            </Button>
          </CardBody>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader title="About" action={<Badge tone="primary">Community Edition</Badge>} />
        <CardBody className="space-y-3 text-sm text-muted-foreground">
          <p>
            The Routeplane CE Console is a self-hosted dashboard for the Community Edition gateway. It talks only to the
            local data plane and holds no secrets beyond your own gateway key.
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
          <p className="text-xs">
            Members & RBAC, SSO & SCIM, and entitlement management are Routeplane Enterprise features handled by the
            control plane. The Community Edition is single-tenant and key-authenticated.
          </p>
        </CardBody>
      </Card>
    </>
  );
}
