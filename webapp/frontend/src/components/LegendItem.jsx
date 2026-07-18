import { useEffect, useRef, useState } from "react";

/**
 * A legend entry (colored dot + label) with a clickable "i" that opens a
 * detailed explanation popover. Closes on click-outside or a second click.
 */
export default function LegendItem({ dotClass, label, title, children, className = "", ariaLabel }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  return (
    <span className="legend" ref={ref}>
      {dotClass && <i className={`dot ${dotClass}`} />}
      <span className={`legend-text ${className}`}>
        {label}
        <button
          className={`legend-info ${open ? "on" : ""}`}
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={ariaLabel || (typeof label === "string" ? `What is ${label}?` : "More information")}
        >
          i
        </button>
      </span>
      {open && (
        <div className="legend-pop">
          <span className="legend-pop-title">{title}</span>
          <div className="legend-pop-body">{children}</div>
        </div>
      )}
    </span>
  );
}
