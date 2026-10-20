import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartCard, ChartTooltip, Legend } from "@/components/analytics/ChartCard";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useCssVars } from "@/hooks/useCssVars";
import { CLASSES, duration, formatDay, percent, STEP_LABELS } from "@/lib/format";

const VARS = [
  "--chart-normal",
  "--chart-pneumonia",
  "--chart-tuberculosis",
  "--chart-series",
  "--chart-grid",
  "--chart-axis",
  "--chart-ink",
  "--card",
];
const CLASS_VAR = { Normal: "--chart-normal", Pneumonia: "--chart-pneumonia", Tuberculosis: "--chart-tuberculosis" };

function axisProps(c) {
  return {
    stroke: c["--chart-axis"],
    tick: { fill: c["--chart-ink"], fontSize: 11 },
    tickLine: false,
  };
}

/** Predictions per day, stacked by class (oldest on the left). */
export function VolumeChart({ daily }) {
  const c = useCssVars(VARS);
  const data = [...daily].reverse().map((d) => ({ day: d.day, total: d.predictions, ...d.by_label }));
  const lastIndex = (row) => CLASSES.filter((k) => row[k]).at(-1); // round only the top segment

  return (
    <ChartCard
      title="Predictions per day"
      description="Stacked by the model's most likely class."
      legend={<Legend items={CLASSES.map((k) => ({ label: k, color: c[CLASS_VAR[k]] }))} />}
      chart={
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} margin={{ top: 8, right: 4, left: -16, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke={c["--chart-grid"]} />
            <XAxis dataKey="day" tickFormatter={formatDay} {...axisProps(c)} minTickGap={16} />
            <YAxis allowDecimals={false} {...axisProps(c)} axisLine={false} width={44} />
            <Tooltip
              cursor={{ fill: c["--chart-grid"], opacity: 0.5 }}
              content={<ChartTooltip formatLabel={formatDay} />}
            />
            {CLASSES.map((k) => (
              <Bar
                key={k}
                dataKey={k}
                name={k}
                stackId="day"
                fill={c[CLASS_VAR[k]]}
                stroke={c["--card"]}
                strokeWidth={2}
                maxBarSize={24}
                shape={(props) => {
                  const { x, y, width, height, fill, stroke, payload } = props;
                  if (!height) return null;
                  const r = lastIndex(payload) === k ? Math.min(4, width / 2, height) : 0;
                  const path = `M${x},${y + height} V${y + r} Q${x},${y} ${x + r},${y} H${x + width - r} Q${x + width},${y} ${x + width},${y + r} V${y + height} Z`;
                  return <path d={path} fill={fill} stroke={stroke} strokeWidth={2} />;
                }}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      }
      table={
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Day (UTC)</TableHead>
              {CLASSES.map((k) => (
                <TableHead key={k} className="text-right">
                  {k}
                </TableHead>
              ))}
              <TableHead className="text-right">Total</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {daily.map((d) => (
              <TableRow key={d.day}>
                <TableCell>{d.day}</TableCell>
                {CLASSES.map((k) => (
                  <TableCell key={k} className="tabular text-right">
                    {d.by_label[k] ?? 0}
                  </TableCell>
                ))}
                <TableCell className="tabular text-right font-medium">{d.predictions}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      }
    />
  );
}

/** Share of each day's predictions that were referred for clinician review. */
export function ReferralChart({ daily }) {
  const c = useCssVars(VARS);
  const data = [...daily].reverse().map((d) => ({ day: d.day, rate: d.referral_rate, referred: d.referred, n: d.predictions }));
  return (
    <ChartCard
      title="Referral rate"
      description="Share of predictions below the confidence threshold, per day."
      chart={
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke={c["--chart-grid"]} />
            <XAxis dataKey="day" tickFormatter={formatDay} {...axisProps(c)} minTickGap={16} />
            <YAxis
              domain={[0, (max) => Math.min(1, Math.max(0.1, Math.ceil(max * 10) / 10))]}
              tickFormatter={(v) => percent(v, 0)}
              {...axisProps(c)}
              axisLine={false}
              width={48}
            />
            <Tooltip
              cursor={{ stroke: c["--chart-axis"] }}
              content={
                <ChartTooltip
                  formatLabel={formatDay}
                  formatValue={(v, p) => `${percent(v)} (${p.payload.referred} of ${p.payload.n})`}
                />
              }
            />
            <Line
              dataKey="rate"
              name="Referred"
              stroke={c["--chart-series"]}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              dot={data.length <= 14 ? { r: 4, fill: c["--chart-series"], stroke: c["--card"], strokeWidth: 2 } : false}
              activeDot={{ r: 5, fill: c["--chart-series"], stroke: c["--card"], strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      }
      table={
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Day (UTC)</TableHead>
              <TableHead className="text-right">Referred</TableHead>
              <TableHead className="text-right">Predictions</TableHead>
              <TableHead className="text-right">Rate</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {daily.map((d) => (
              <TableRow key={d.day}>
                <TableCell>{d.day}</TableCell>
                <TableCell className="tabular text-right">{d.referred}</TableCell>
                <TableCell className="tabular text-right">{d.predictions}</TableCell>
                <TableCell className="tabular text-right font-medium">{percent(d.referral_rate)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      }
    />
  );
}


/** Average latency per pipeline step, slowest first; the value sits at each bar's tip. */
export function StepLatencyChart({ steps }) {
  const c = useCssVars(VARS);
  const data = Object.entries(steps)
    .map(([name, s]) => ({ name: STEP_LABELS[name] ?? name, ms: s.avg_latency_ms, count: s.count, errors: s.error_rate }))
    .sort((a, b) => b.ms - a.ms);
  const height = Math.max(160, data.length * 34 + 24);

  return (
    <ChartCard
      title="Average time per step"
      description="From your recorded traces. Totals include the steps inside them."
      chart={
        <ResponsiveContainer width="100%" height={height}>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 64, left: 8, bottom: 0 }}>
            <CartesianGrid horizontal={false} stroke={c["--chart-grid"]} />
            <XAxis type="number" tickFormatter={duration} {...axisProps(c)} />
            <YAxis type="category" dataKey="name" width={124} {...axisProps(c)} axisLine={false} />
            <Tooltip
              cursor={{ fill: c["--chart-grid"], opacity: 0.5 }}
              content={<ChartTooltip formatValue={(v, p) => `${duration(v)} · ${p.payload.count} runs`} />}
            />
            <Bar dataKey="ms" name="Average" fill={c["--chart-series"]} maxBarSize={18} radius={[0, 4, 4, 0]}>
              <LabelList dataKey="ms" position="right" formatter={duration} fill={c["--chart-ink"]} fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      }
      table={
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Step</TableHead>
              <TableHead className="text-right">Runs</TableHead>
              <TableHead className="text-right">Average</TableHead>
              <TableHead className="text-right">Errors</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((d) => (
              <TableRow key={d.name}>
                <TableCell>{d.name}</TableCell>
                <TableCell className="tabular text-right">{d.count}</TableCell>
                <TableCell className="tabular text-right font-medium">{duration(d.ms)}</TableCell>
                <TableCell className="tabular text-right">{percent(d.errors)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      }
    />
  );
}
