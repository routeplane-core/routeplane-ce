import { useNavigate } from "react-router-dom";
import { Compass } from "lucide-react";
import { Card, CardBody } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/states";

export function NotFound() {
  const navigate = useNavigate();
  return (
    <Card className="mx-auto max-w-lg">
      <CardBody>
        <EmptyState
          icon={Compass}
          title="Page not found"
          description="This page doesn't exist in the Community Edition Console."
          action={
            <Button variant="outline" onClick={() => navigate("/")}>
              Back to Overview
            </Button>
          }
        />
      </CardBody>
    </Card>
  );
}
