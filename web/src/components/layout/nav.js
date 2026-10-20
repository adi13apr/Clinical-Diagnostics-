import { Activity, BarChart3, History, ScanLine } from "lucide-react";

/** One list drives the sidebar and the mobile bottom nav. */
export const NAV_ITEMS = [
  { to: "/", label: "Diagnose", icon: ScanLine, end: true },
  { to: "/history", label: "History", icon: History },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/traces", label: "Traces", icon: Activity },
];
