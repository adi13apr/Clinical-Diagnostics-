import { motion } from "framer-motion";
import { ImageUp } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

const ACCEPTED = ["image/jpeg", "image/png"];
const MAX_UPLOAD_MB = 10; // matches the API's default MAX_UPLOAD_MB

/** Client-side check, so obvious mistakes fail instantly; the server checks again. */
function validateFile(file) {
  if (!file) return "No file selected.";
  if (!ACCEPTED.includes(file.type)) return "Please choose a JPEG or PNG image.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `That file is over ${MAX_UPLOAD_MB} MB.`;
  return null;
}

/**
 * Drag-and-drop target that is also a real button: Tab to it, press Enter or Space to browse.
 * @param {{onFile: (file: File) => void, onError: (message: string) => void, compact?: boolean, disabled?: boolean}} props
 */
export function UploadDropzone({ onFile, onError, compact = false, disabled = false }) {
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);

  function take(files) {
    const file = files?.[0];
    if (!file) return;
    const problem = validateFile(file);
    if (problem) onError(problem);
    else onFile(file);
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) take(e.dataTransfer.files);
      }}
    >
      <button
        type="button"
        disabled={disabled}
        onClick={() => input.current?.click()}
        aria-describedby="upload-hint"
        className={cn(
          "group flex w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed bg-card text-center transition-colors outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:opacity-60",
          compact ? "gap-2 px-4 py-6" : "gap-4 px-6 py-14 sm:py-20",
          dragging ? "border-primary bg-accent" : "border-input hover:border-primary/60 hover:bg-accent/40",
        )}
      >
        <span
          className={cn(
            "flex items-center justify-center rounded-full bg-accent text-accent-foreground transition-transform group-hover:scale-105",
            compact ? "size-10" : "size-16",
          )}
        >
          <ImageUp className={compact ? "size-5" : "size-8"} aria-hidden="true" />
        </span>
        <span className={cn("font-semibold", compact ? "text-sm" : "text-lg")}>
          {dragging ? "Drop to analyse" : compact ? "Analyse another X-ray" : "Drop a chest X-ray here"}
        </span>
        <span id="upload-hint" className="text-sm text-muted-foreground">
          or <span className="font-medium text-primary underline-offset-4 group-hover:underline">browse files</span> ·
          JPEG or PNG, up to {MAX_UPLOAD_MB} MB
        </span>
      </button>
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png"
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          take(e.target.files);
          e.target.value = ""; // choosing the same file again still triggers a change
        }}
      />
    </motion.div>
  );
}
