import { cn } from "@/lib/utils";

// The Routeplane brand mark — a hub-and-spoke "routing" glyph (navy → royal
// blue) on the brand-dark tile, matching public/favicon.svg and the marketing
// site. Rendered inline so it scales crisply and never needs a network fetch.
export function LogoMark({ size = 32, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      role="img"
      aria-label="Routeplane"
      className={cn("shrink-0", className)}
    >
      <defs>
        <linearGradient id="rp-hub" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#2563EB" />
          <stop offset="50%" stopColor="#1D4ED8" />
          <stop offset="100%" stopColor="#1E3A8A" />
        </linearGradient>
      </defs>
      <rect width="80" height="80" rx="18" fill="#0B1120" />
      <g transform="scale(0.78) translate(11 11)">
        <line x1="40" y1="40" x2="6" y2="12" stroke="#60A5FA" strokeWidth="5" strokeLinecap="round" />
        <line x1="40" y1="40" x2="6" y2="68" stroke="#60A5FA" strokeWidth="5" strokeLinecap="round" />
        <line x1="40" y1="40" x2="74" y2="12" stroke="#1D4ED8" strokeWidth="5" strokeLinecap="round" />
        <line x1="40" y1="40" x2="74" y2="68" stroke="#1D4ED8" strokeWidth="5" strokeLinecap="round" />
        <circle cx="6" cy="12" r="6.5" fill="#2563EB" />
        <circle cx="6" cy="68" r="6.5" fill="#2563EB" />
        <circle cx="74" cy="12" r="6.5" fill="#1D4ED8" />
        <circle cx="74" cy="68" r="6.5" fill="#1D4ED8" />
        <circle cx="40" cy="40" r="16" fill="url(#rp-hub)" />
      </g>
    </svg>
  );
}

/** Mark + wordmark lockup. `sublabel` renders the product line (e.g. "Console"). */
export function BrandLockup({ sublabel, size = 32 }: { sublabel?: string; size?: number }) {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark size={size} />
      <span className="leading-tight">
        <span className="block text-sm font-semibold tracking-tight">
          routeplane<span className="font-medium text-muted-foreground">.ai</span>
        </span>
        {sublabel && <span className="block text-2xs uppercase tracking-wider text-muted-foreground">{sublabel}</span>}
      </span>
    </span>
  );
}
