import { useMemo, useState, type ReactNode } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

export interface Column<T> {
  key: string;
  header: ReactNode;
  cell: (row: T) => ReactNode;
  /** Provide to make the column sortable. */
  sortValue?: (row: T) => string | number;
  align?: "left" | "right" | "center";
  /** Tailwind width class, e.g. "w-40". */
  width?: string;
  className?: string;
  headerClassName?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  getRowId: (row: T) => string;
  onRowClick?: (row: T) => void;
  /** Initial sort column key. */
  defaultSort?: { key: string; dir: "asc" | "desc" };
  empty?: ReactNode;
  className?: string;
  dense?: boolean;
}

export function DataTable<T>({
  columns,
  rows,
  getRowId,
  onRowClick,
  defaultSort,
  empty,
  className,
  dense,
}: DataTableProps<T>) {
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(defaultSort ?? null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = col.sortValue!(a);
      const bv = col.sortValue!(b);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    });
  }, [rows, sort, columns]);

  const toggleSort = (key: string) => {
    setSort((s) =>
      s?.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" },
    );
  };

  const alignCls = (a?: string) =>
    a === "right" ? "text-right" : a === "center" ? "text-center" : "text-left";

  return (
    <div className={cn("overflow-x-auto scroll-thin", className)}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-xs text-muted-foreground">
            {columns.map((c) => (
              <th
                key={c.key}
                className={cn(
                  "whitespace-nowrap px-4 py-2.5 font-medium",
                  alignCls(c.align),
                  c.width,
                  c.headerClassName,
                  c.sortValue && "cursor-pointer select-none hover:text-foreground",
                )}
                onClick={c.sortValue ? () => toggleSort(c.key) : undefined}
              >
                <span className={cn("inline-flex items-center gap-1", c.align === "right" && "flex-row-reverse")}>
                  {c.header}
                  {c.sortValue && sort?.key === c.key && (
                    sort.dir === "asc" ? <ChevronUp size={12} /> : <ChevronDown size={12} />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-4">
                {empty ?? <div className="py-12 text-center text-sm text-muted-foreground">No rows.</div>}
              </td>
            </tr>
          ) : (
            sorted.map((row) => (
              <tr
                key={getRowId(row)}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={cn(
                  "border-b last:border-0",
                  onRowClick && "cursor-pointer hover:bg-muted/50",
                )}
              >
                {columns.map((c) => (
                  <td
                    key={c.key}
                    className={cn(dense ? "px-4 py-2" : "px-4 py-2.5", alignCls(c.align), c.className)}
                  >
                    {c.cell(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
