import { motion } from "framer-motion";
import { Eye, EyeOff, Loader2, RotateCw, TriangleAlert } from "lucide-react";
import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { useGradcamMap } from "@/hooks/queries";
import { drawCam, JET_GRADIENT } from "@/lib/colormap";
import { percent } from "@/lib/format";

const ZONE_LABEL = (zone) => `patient's ${zone}`; // "right upper zone": patient side, assuming a PA view

function RegionSummary({ regions }) {
  if (!regions || regions.focus === "none") {
    return <p className="text-sm text-muted-foreground">{regions?.note || "No clear area of influence."}</p>;
  }
  const share = regions.region_shares?.[regions.peak_region];
  return (
    <div className="space-y-2 text-sm">
      <p>
        <span className="text-muted-foreground">Strongest area: </span>
        <span className="font-medium">{ZONE_LABEL(regions.peak_region)}</span>
        <span className="tabular text-muted-foreground"> ({percent(share, 0)} of the weight)</span>
      </p>
      <p className="text-muted-foreground">
        {regions.focus === "focal"
          ? "Attention is concentrated in one area."
          : "Attention is spread out, so no single area explains the result."}
      </p>
      {regions.border_attention && (
        <p className="flex gap-2 rounded-lg bg-warning-soft p-2.5 text-warning">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <span>
            Over half the weight is on the image border ({percent(regions.border_share, 0)}), not the lungs. The model may
            be reacting to edges or markers; treat this result with extra caution.
          </span>
        </p>
      )}
    </div>
  );
}

/**
 * The uploaded X-ray with the Grad-CAM heatmap drawn on top. The map is stretched over the whole
 * image, exactly as the model saw it (resized to a square), and the slider sets its opacity.
 */
export function GradcamViewer({ previewUrl, traceId, opacity, onOpacityChange, scanning = false }) {
  const canvas = useRef(null);
  const map = useGradcamMap(scanning ? null : traceId);
  const cam = map.data?.cam;

  useEffect(() => {
    if (cam && canvas.current) drawCam(canvas.current, cam);
  }, [cam]);

  const visible = opacity > 0;
  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="text-base">Where the model looked</CardTitle>
        <CardDescription>
          Grad-CAM heatmap for {map.data?.explained_class ?? "the predicted class"}. Red areas influenced the result
          most. It is a coarse 7×7 map, not a lung outline.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex justify-center rounded-xl bg-black/90 p-2 dark:bg-black">
          <div className="relative w-fit max-w-full overflow-hidden rounded-lg">
            <img
              src={previewUrl}
              alt="Uploaded chest X-ray"
              className="block h-auto max-h-[26rem] w-auto max-w-full select-none"
              draggable={false}
            />
            <canvas
              ref={canvas}
              width={224}
              height={224}
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 h-full w-full transition-opacity duration-200"
              style={{ opacity: cam ? opacity / 100 : 0, mixBlendMode: "normal" }}
            />
            {scanning && (
              <motion.div
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-[color-mix(in_oklab,var(--primary)_45%,transparent)] to-transparent"
                initial={{ top: "-6rem" }}
                animate={{ top: "100%" }}
                transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
              />
            )}
            {map.isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/30" role="status">
                <span className="inline-flex items-center gap-2 rounded-full bg-background/90 px-3 py-1.5 text-xs font-medium">
                  <Loader2 className="size-3.5 animate-spin" aria-hidden="true" /> Computing heatmap…
                </span>
              </div>
            )}
          </div>
        </div>

        {scanning ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
            <Loader2 className="size-4 animate-spin" aria-hidden="true" /> Analysing the image…
          </p>
        ) : !traceId ? (
          <p className="text-sm text-muted-foreground">The heatmap appears after a successful analysis.</p>
        ) : map.isError ? (
          <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-destructive/30 p-3 text-sm">
            <span>Heatmap unavailable: {map.error.message}</span>
            <Button size="sm" variant="outline" onClick={() => map.refetch()}>
              <RotateCw aria-hidden="true" /> Retry
            </Button>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="icon-sm"
                onClick={() => onOpacityChange(visible ? 0 : 45)}
                aria-label={visible ? "Hide heatmap" : "Show heatmap"}
                disabled={!cam}
              >
                {visible ? <EyeOff aria-hidden="true" /> : <Eye aria-hidden="true" />}
              </Button>
              <div className="flex-1">
                <div className="mb-2 flex justify-between text-xs text-muted-foreground">
                  <span id="opacity-label">Heatmap opacity</span>
                  <span className="tabular">{opacity}%</span>
                </div>
                <Slider
                  value={[opacity]}
                  onValueChange={([v]) => onOpacityChange(v)}
                  min={0}
                  max={100}
                  step={5}
                  disabled={!cam}
                  aria-labelledby="opacity-label"
                />
              </div>
            </div>
            <div>
              <div className="h-2 rounded-full" style={{ background: JET_GRADIENT }} aria-hidden="true" />
              <div className="mt-1 flex justify-between text-xs text-muted-foreground">
                <span>Low influence</span>
                <span>High influence</span>
              </div>
            </div>
            {map.data && <RegionSummary regions={map.data.regions} />}
          </>
        )}
      </CardContent>
    </Card>
  );
}
